import json
import re
import os
import math
import argparse
from collections import Counter

RES_PATH = "<PATH_TO_RESULTS_JSONL>" 

def extract_answer(text):
    if not text:
        return "", False
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").strip()
    
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    all_boxes = re.findall(boxed_pattern, text)
    if all_boxes:
        return all_boxes[-1], True
    
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip(), True
    
    marker = "the answer is"
    if marker in text.lower():
        pos = text.lower().rfind(marker)
        after_text = text[pos + len(marker):].strip()
        after_text = re.sub(r"^[:\s]+", "", after_text)
        return after_text.split('\n')[0].split('$')[0].strip(), True

    tail = text[-50:].strip()
    nums = re.findall(r"(-?\d+[\./\d]*|\\sqrt\{\d+\}|\(-?\d+.*?\))", tail)
    if nums:
        return nums[-1], False
    return "", False

def normalize_math(string):
    if not string: return ""
    string = str(string).lower().strip()
    
    string = string.replace("</reasoning>", "").replace("</answer>", "").replace("<answer>", "")
    string = string.replace("...", "").replace("cannot be determined", "")
    
    string = re.sub(r"([a-z]+|\\theta|\\alpha|\\pi)\s*=\s*", "", string)
    string = re.sub(r"\\text\{([^}]*)\}", r"\1", string)
    string = re.sub(r"\\(mathbf|mathrm|bold|unit|mbox|operatorname|mathrm)\{([^}]*)\}", r"\2", string)
    string = re.sub(r"\\(d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"\2/\3", string)
    string = string.replace("\\!", "").replace("\\ ", "").replace("{", "").replace("}", "")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    
    units_pattern = r"(units?|cm\^2|cm|inches|inch|square|degrees?|radians?|miles?|per|hour|cents?)"
    string = re.sub(units_pattern, "", string)
    string = string.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "").replace("\\degree", "")
    string = string.replace("\\pi", "pi")
    string = re.sub(r"(\d),(\d{3})", r"\1\2", string) 
    string = string.rstrip(".:,; ").replace(" ", "")
    
    if "=" in string:
        string = string.split("=")[-1]
        
    return string

def is_equiv(pred, gold):
    if not pred: return False
    p, g = normalize_math(pred), normalize_math(gold)
    if p == g: return True
    
    if "=" in pred:
        if normalize_math(pred.split("=")[-1]) == g:
            return True
            
    try:
        def to_float(s):
            if '/' in s and s.count('/') == 1:
                parts = s.split('/')
                return float(parts[0]) / float(parts[1])
            if '_' in s: s = s.split('_')[0]
            return float(s)
        return math.isclose(to_float(p), to_float(g), rel_tol=1e-4)
    except:
        p_fuzzy = re.sub(r"[^a-z0-9/,\-]", "", p)
        g_fuzzy = re.sub(r"[^a-z0-9/,\-]", "", g)
        return p_fuzzy == g_fuzzy if p_fuzzy else False

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
        
        voted_correct_count = 0        
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
                ground_truth = str(item.get("target", doc.get("answer", "")))
                
                current_nfe = item.get("nfe", 0)
                nfe_list.append(current_nfe)
                current_svf = item.get("svf_calls", 0)
                svf_list.append(current_svf)
                
                ans_stats = {} 
                trajectories = item.get("all_trajectories", [])
                
                for traj in trajectories:
                    raw_text = traj.get("resp", "")
                    score = traj.get("score", 0) 
                    
                    extracted, _ = extract_answer(raw_text)
                    if not extracted: continue
                    
                    norm = normalize_math(extracted)
                    if norm not in ans_stats:
                        ans_stats[norm] = {
                            "count": 0, 
                            "max_score": -float('inf'), 
                            "total_weight": 0.0,
                            "original": extracted
                        }
                    
                    ans_stats[norm]["count"] += 1
                    if score > ans_stats[norm]["max_score"]:
                        ans_stats[norm]["max_score"] = score
                    
                    try:
                        weight = math.exp(score)
                    except OverflowError:
                        weight = float('inf')
                    ans_stats[norm]["total_weight"] += weight

                if not ans_stats:
                    best_pred = ""
                else:
                    sorted_norms = sorted(
                        ans_stats.keys(), 
                        key=lambda x: (ans_stats[x]["total_weight"], ans_stats[x]["max_score"], ans_stats[x]["count"]), 
                        reverse=True
                    )
                    best_norm = sorted_norms[0]
                    best_pred = ans_stats[best_norm]["original"]
                
                is_voted_correct = False
                if best_pred and is_equiv(best_pred, ground_truth):
                    voted_correct_count += 1
                    is_voted_correct = True
                
                total_count += 1
                
                detailed_results.append({
                    "question": doc.get("problem", "N/A"),
                    "final_voted_answer": best_pred,
                    "ground_truth": ground_truth,
                    "is_voted_correct": is_voted_correct,
                    "nfe": current_nfe,
                    "svf_calls": current_svf
                })

        accuracy = (voted_correct_count / total_count * 100) if total_count > 0 else 0
        
        avg_nfe = sum(nfe_list) / len(nfe_list) if nfe_list else 0
        avg_svf = sum(svf_list) / len(svf_list) if svf_list else 0
        
        print(f"---  Accuracy : {accuracy:.2f}% | NFE: {avg_nfe:.1f} | SVF: {avg_svf:.1f} ---")

        output_name = f"eval_voted_{os.path.basename(file_path).replace('.jsonl', '.json')}"
        output_path = os.path.join(os.path.dirname(file_path), output_name)
        
        final_report = {
            "summary": {
                "Accuracy": f"{accuracy:.2f}%", 
                "correct_voted_count": voted_correct_count,
                "total": total_count, 
                "avg_nfe": avg_nfe,
                "avg_svf": avg_svf
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