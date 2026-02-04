import json
import re
import os
import math
import argparse
from collections import Counter

RES_PATH = "<PATH_TO_RESULTS_JSONL>"

def last_boxed_only_string(string):
    if not string: return None
    idx = max(string.rfind("\\boxed"), string.rfind("\\fbox"))
    if idx < 0: return None
    
    if "\\boxed " in string[idx:idx+8] and "{" not in string[idx:idx+8]:
        return "\\boxed " + string[idx:].split("\\boxed ")[-1].split("$")[0].strip()

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx else None

def remove_boxed(s):
    if not s: return None
    if "\\boxed " in s: return s[len("\\boxed ") :]
    if "\\boxed{" in s and s.endswith("}"): return s[len("\\boxed{") : -1]
    if "\\fbox{" in s and s.endswith("}"): return s[len("\\fbox{") : -1]
    return s

def strip_string(string):
    if string is None: return ""
    string = str(string).strip()
    while re.search(r"(\d),(\d{3})", string):
        string = re.sub(r"(\d),(\d{3})", r"\1\2", string)
    
    string = string.replace("\n", "").replace("\\!", "")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "").replace("\%", "")
    
    if "=" in string and len(string.split("=")[0]) <= 5:
        string = string.split("=")[1].strip()
        
    string = string.replace(" ", "")
    string = string.rstrip(".")
    return string

def normalize_to_number(s):
    s_clean = strip_string(s)
    try:
        if '/' in s_clean and len(s_clean.split('/')) == 2:
            parts = s_clean.split('/')
            return float(parts[0]) / float(parts[1])
        return float(s_clean)
    except:
        return s_clean

def extract_answer_gsm8k_debug(text):
    if not text: return "", "empty"
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").strip()

    boxed = last_boxed_only_string(text)
    if boxed:
        ans = remove_boxed(boxed)
        if ans: 
            return strip_string(ans), "boxed"
    
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        return strip_string(tag_match.group(1)), "xml_tag"
    
    last_text = text[-200:] if len(text) > 200 else text
    marker = "the answer is"
    if marker in last_text.lower():
        idx = last_text.lower().rfind(marker)
        after = last_text[idx + len(marker):].strip()
        after = re.split(r"[.\n]", after)[0]
        after = after.replace(":", "").replace("$", "").strip()
        return strip_string(after), "text_marker"

    tail = text[-50:]
    nums = re.findall(r"(?<!\d)-?\d+\.?\d*(?!\d)", tail)
    if nums:
        return strip_string(nums[-1]), "regex_last_num"
    
    return "", "failed"

def extract_gold_gsm8k(target_str):
    if "####" in target_str:
        return strip_string(target_str.split("####")[-1])
    return strip_string(target_str)

def is_equiv(pred, gold):
    p_val = normalize_to_number(pred)
    g_val = normalize_to_number(gold)
    
    if isinstance(p_val, float) and isinstance(g_val, float):
        return math.isclose(p_val, g_val, rel_tol=1e-4)
    return str(p_val) == str(g_val)

def run_evaluation(target_path):
    jsonl_files = []
    if os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.endswith(".jsonl") and not file.startswith("eval_voted_"):
                    jsonl_files.append(os.path.join(root, file))
    else:
        jsonl_files = [target_path]

    for file_path in jsonl_files:
        print(f">>> 正在评测: {file_path}")
        detailed_results = []
        
        correct_voted_count = 0
        correct_any_count = 0 
        total_count = 0
        nfe_list = []
        svf_list = [] 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                except:
                    continue
                    
                doc = item.get("doc", {})
                ground_truth = extract_gold_gsm8k(str(item.get("target", "")))
                
                total_nfe_item = item.get("nfe", 0)
                nfe_list.append(total_nfe_item)
                svf_list.append(item.get("svf_calls", 0))
                
                trajectories = item.get("all_trajectories", [])
                if not trajectories:
                    resps = item.get("resps", [])
                    for r in resps:
                        text = r[0] if isinstance(r, list) else r
                        trajectories.append({"resp": text, "score": 0.0})

                parsed_paths = []
                traj_debug_info = [] 

                for idx, traj in enumerate(trajectories):
                    raw_text = traj.get("resp", "")
                    score = traj.get("score", 0.0)
                    
                    extracted, method = extract_answer_gsm8k_debug(raw_text)
                    
                    is_correct_single = False
                    if extracted:
                        is_correct_single = is_equiv(extracted, ground_truth)
                        val_key = normalize_to_number(extracted)
                        
                        parsed_paths.append({
                            "original_text": extracted,
                            "val_key": val_key,
                            "score": score,
                            "method": method
                        })
                    
                    traj_debug_info.append({
                        "id": idx,
                        "extracted": extracted,
                        "score": score,
                        "is_correct": is_correct_single,
                        "extract_method": method
                    })

                if not parsed_paths:
                    detailed_results.append({
                        "question": doc.get("question", "N/A"),
                        "final_voted_answer": "",
                        "ground_truth": ground_truth,
                        "is_voted_correct": False,
                        "trajectory_details": traj_debug_info,
                        "nfe": total_nfe_item,
                        "svf_calls": item.get("svf_calls", 0)
                    })
                    total_count += 1
                    continue

                has_correct = any(p['score'] > -999 and is_equiv(p['original_text'], ground_truth) for p in parsed_paths)
                if has_correct:
                    correct_any_count += 1

                parsed_paths.sort(key=lambda x: x['score'], reverse=True)
                top_k_count = max(1, int(len(parsed_paths) * 0.6))
                voting_candidates = parsed_paths[:top_k_count]
                
                ans_stats = {}
                for p in voting_candidates:
                    k = p['val_key']
                    if k not in ans_stats:
                        ans_stats[k] = {
                            "total_weight": 0.0,
                            "count": 0,
                            "max_score": -float('inf'),
                            "best_repr": p['original_text']
                        }
                    
                    try:
                        weight = math.exp(p['score'])
                    except OverflowError:
                        weight = float('inf')
                        
                    ans_stats[k]["total_weight"] += weight
                    ans_stats[k]["count"] += 1
                    if p['score'] > ans_stats[k]["max_score"]:
                        ans_stats[k]["max_score"] = p['score']
                        ans_stats[k]["best_repr"] = p['original_text']

                sorted_answers = sorted(
                    ans_stats.items(), 
                    key=lambda x: (x[1]["total_weight"], x[1]["max_score"]), 
                    reverse=True
                )
                
                best_pred = str(sorted_answers[0][1]["best_repr"])
                is_voted_correct = is_equiv(best_pred, ground_truth)
                if is_voted_correct:
                    correct_voted_count += 1
                
                vote_summary = []
                for val, info in sorted_answers:
                    vote_summary.append({
                        "answer": str(val),
                        "count": info["count"],
                        "total_weight": info["total_weight"],
                        "is_correct": is_equiv(str(val), ground_truth)
                    })

                total_count += 1
                
                detailed_results.append({
                    "question": doc.get("question", "N/A"),
                    "final_voted_answer": best_pred,
                    "ground_truth": ground_truth,
                    "is_voted_correct": is_voted_correct,
                    "vote_stats": vote_summary, 
                    "trajectory_details": traj_debug_info,
                    "nfe": total_nfe_item,
                    "svf_calls": item.get("svf_calls", 0)
                })

        accuracy = (correct_voted_count / total_count * 100) if total_count > 0 else 0
        pass_at_k = (correct_any_count / total_count * 100) if total_count > 0 else 0
        avg_nfe = int(round(sum(nfe_list) / len(nfe_list))) if nfe_list else 0
        avg_svf = int(round(sum(svf_list) / len(svf_list))) if svf_list else 0
        
        print(f"--- Accuracy: {accuracy:.2f}% | NFE: {avg_nfe} | SVF: {avg_svf} ---")

        output_name = f"eval_voted_{os.path.basename(file_path).replace('.jsonl', '.json')}"
        output_path = os.path.join(os.path.dirname(file_path), output_name)
        
        final_report = {
            "summary": {
                "accuracy": f"{accuracy:.2f}%", 
                "correct_voted": correct_voted_count,
                "total": total_count, 
                "nfe": avg_nfe,
                "svf_calls": avg_svf
            },
            "details": detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_report, out_f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--res_path", type=str, default=RES_PATH)
    args = parser.parse_args()
    run_evaluation(args.res_path)