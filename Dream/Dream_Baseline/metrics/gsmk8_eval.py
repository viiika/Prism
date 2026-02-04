import json
import re
import os
import glob
import math
import argparse
from collections import Counter


RES_PATH = "<PATH_TO_RESULTS_JSONL>"


def last_boxed_only_string(string):
    if not string: return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0: return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
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
    if "=" in string and len(string.split("=")[0]) <= 3:
        string = string.split("=")[1].strip()
    string = string.replace(" ", "")
    return string

def extract_answer_gsm8k(text):
    if not text: return ""
    boxed = last_boxed_only_string(text)
    if boxed:
        ans = remove_boxed(boxed)
        if ans: return strip_string(ans)
    
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        return strip_string(tag_match.group(1))
    
    nums = re.findall(r"-?\d+\.?\d*", text[-50:])
    if nums:
        return strip_string(nums[-1])
    
    return ""

def extract_gold_gsm8k(target_str):
    if "####" in target_str:
        return strip_string(target_str.split("####")[-1])
    return strip_string(target_str)

def is_equiv(pred, gold):
    p = strip_string(pred)
    g = strip_string(gold)
    try:
        return math.isclose(float(p), float(g), rel_tol=1e-4)
    except:
        return p == g

def run_evaluation(target_path):
    if os.path.isdir(target_path):
        jsonl_files = glob.glob(os.path.join(target_path, "*.jsonl"))
    else:
        jsonl_files = [target_path]

    for file_path in jsonl_files:
        print(f">>> 正在评测: {file_path}")
        detailed_results = []
        correct_count = 0
        total_count = 0
        nfe_list = []
        svf_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                doc = item.get("doc", {})
                
                ground_truth = extract_gold_gsm8k(str(item.get("target", "")))
                nfe_list.append(item.get("nfe", 0))
                svf_list.append(item.get("svf_calls", 0))
                
                ans_stats = {}
                
                trajectories = item.get("all_trajectories", [])
                if not trajectories:
                    resps = item.get("resps", [])
                    for r in resps:
                        text = r[0] if isinstance(r, list) else r
                        trajectories.append({"resp": text, "score": 0.0})

                for traj in trajectories:
                    raw_text = traj.get("resp", "")
                    score = traj.get("score", -float('inf'))
                    extracted = extract_answer_gsm8k(raw_text)
                    
                    if not extracted: continue
                    
                    norm = strip_string(extracted)
                    if norm not in ans_stats:
                        ans_stats[norm] = {"count": 0, "max_score": -float('inf'), "original": extracted}
                    
                    ans_stats[norm]["count"] += 1
                    if score > ans_stats[norm]["max_score"]:
                        ans_stats[norm]["max_score"] = score
                        ans_stats[norm]["original"] = extracted

                if not ans_stats:
                    best_pred = ""
                else:
                    sorted_norms = sorted(
                        ans_stats.keys(), 
                        key=lambda x: (ans_stats[x]["count"], ans_stats[x]["max_score"]), 
                        reverse=True
                    )
                    best_norm = sorted_norms[0]
                    best_pred = ans_stats[best_norm]["original"]
                
                ans_correct = is_equiv(best_pred, ground_truth)
                if ans_correct:
                    correct_count += 1
                total_count += 1
                
                detailed_results.append({
                    "question": doc.get("question", "N/A"),
                    "final_voted_answer": best_pred,
                    "ground_truth": ground_truth,
                    "is_correct": ans_correct,
                    "nfe": item.get("nfe", 0),
                    "svf_calls": item.get("svf_calls", 0)
                })

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        avg_nfe = int(round(sum(nfe_list) / len(nfe_list))) if nfe_list else 0
        avg_svf = int(round(sum(svf_list) / len(svf_list))) if svf_list else 0
        
        print(f"Accuracy: {accuracy:.2f}% | NFE: {avg_nfe} | SVF: {avg_svf} ---")

        output_name = f"eval_voted_{os.path.basename(file_path).replace('.jsonl', '.json')}"
        output_path = os.path.join(os.path.dirname(file_path), output_name)
        
        final_report = {
            "summary": {
                "accuracy": f"{accuracy:.2f}%", 
                "correct": correct_count, 
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