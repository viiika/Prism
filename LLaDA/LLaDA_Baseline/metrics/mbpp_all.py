import os
import json
import ast
import glob
import re
import argparse
from typing import Dict, List, Optional, Set, Tuple
import evaluate as hf_evaluate

RES_PATH = "<PATH_TO_RESULTS_JSONL>"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_python_code(text: str) -> str:
    if not text: return ""
    
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").replace("<|notification_end|>", "")
    
    tag_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_matches:
        for block in tag_matches:
            if "def " in block:
                text = block
                break
        else:
            text = tag_matches[0]

    if "```python" in text:
        blocks = text.split("```python")
        for b in blocks[1:]:
            code = b.split("```")[0].strip()
            if "def " in code: return code
    elif "```" in text:
        blocks = text.split("```")
        for b in blocks[1:]:
            code = b.strip()
            if "def " in code: return code

    lines = text.split('\n')
    cleaned_lines = []
    stop_words = ["Explanation:", "Example:", "Test Case:", "Output:", "Reasoning:"]
    for line in lines:
        if any(sw in line for sw in stop_words): break
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()

def normalize_code_for_voting(code: str) -> str:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        return ast.unparse(tree).strip()
    except:
        return re.sub(r"\s+", "", code)

def run_evaluation(target_path):
    target_path = os.path.abspath(target_path)
    
    if os.path.isdir(target_path):
        search_pattern = os.path.join(target_path, "**/*.jsonl")
        jsonl_files = glob.glob(search_pattern, recursive=True)
        jsonl_files = [f for f in jsonl_files if not os.path.basename(f).startswith("eval_mbpp_")]
    else:
        jsonl_files = [target_path]

    if not jsonl_files:
        print(f"Error: 在路径 {target_path} 及其子目录下未找到任何 .jsonl 文件。")
        return

    try:
        code_eval = hf_evaluate.load("code_eval")
    except:
        print("Error: Could not load code_eval. Ensure 'evaluate' and 'code_eval' are installed.")
        return

    for file_path in jsonl_files:
        print(f"\n>>> 正在评测 MBPP 文件: {file_path}")
        all_candidate_predictions = [] 
        all_voted_predictions = []     
        all_references = []
        detailed_results = []
        nfe_list = []
        svf_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                doc = item.get("doc", {})
                test_list = doc.get("test_list", [])
                test_setup = doc.get("test_setup_code", "")
                full_reference = (test_setup + "\n" + "\n".join(test_list)).strip()
                
                item_nfe = item.get("nfe", 0)
                item_svf = item.get("svf_calls", 0)
                nfe_list.append(item_nfe)
                svf_list.append(item_svf)

                resps = item.get("resps", [])
                trajs = item.get("all_trajectories", [])
                
                candidate_stats = {}
                processed_candidates = []
                
                source_data = trajs if trajs else resps
                for idx, entry in enumerate(source_data):
                    raw_text = entry.get("resp", "") if isinstance(entry, dict) else (entry[0] if isinstance(entry, list) else entry)
                    score = entry.get("score", 0) if isinstance(entry, dict) else 0
                    
                    code = extract_python_code(raw_text)
                    if not code: continue
                    
                    processed_candidates.append(code)
                    
                    try:
                        ast.parse(code)
                        is_valid = True
                    except:
                        is_valid = False
                        
                    norm = normalize_code_for_voting(code)
                    if norm not in candidate_stats:
                        candidate_stats[norm] = {"count": 0, "valid": is_valid, "code": code, "max_score": -float('inf')}
                    candidate_stats[norm]["count"] += 1
                    candidate_stats[norm]["max_score"] = max(candidate_stats[norm]["max_score"], score)

                if not candidate_stats:
                    voted_code = ""
                else:
                    sorted_norms = sorted(
                        candidate_stats.keys(),
                        key=lambda k: (candidate_stats[k]["valid"], candidate_stats[k]["max_score"], candidate_stats[k]["count"]),
                        reverse=True
                    )
                    voted_code = candidate_stats[sorted_norms[0]]["code"]

                all_candidate_predictions.append(processed_candidates if processed_candidates else [""])
                all_voted_predictions.append([voted_code])
                all_references.append(full_reference)
                
                detailed_results.append({
                    "task_id": doc.get("task_id", "N/A"),
                    "voted_code": voted_code,
                    "nfe": item_nfe,
                    "svf_calls": item_svf,
                    "candidates_count": len(processed_candidates)
                })

        if not all_voted_predictions:
            continue

        print(f"正在测试代码 (共 {len(all_voted_predictions)} 题)...")
        res_voted, details_voted = code_eval.compute(references=all_references, predictions=all_voted_predictions, k=[1])
        res_pk, details_pk = code_eval.compute(references=all_references, predictions=all_candidate_predictions, k=[1])

        acc_voted = res_voted.get("pass@1", 0.0) * 100
        acc_pk = res_pk.get("pass@1", 0.0) * 100
        avg_nfe = int(round(sum(nfe_list) / len(nfe_list))) if nfe_list else 0
        avg_svf = int(round(sum(svf_list) / len(svf_list))) if svf_list else 0

        print(f"--- Pass@1: {acc_voted:.2f}% | NFE: {avg_nfe} | SVF: {avg_svf} ---")

        for i, detail in enumerate(detailed_results):
            detail["is_voted_correct"] = details_voted.get(i, [[0, {"passed": False}]])[0][1]["passed"]

        file_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        output_name = f"eval_mbpp_{base_name.replace('.jsonl', '.json')}"
        output_path = os.path.join(file_dir, output_name)
        
        final_report = {
            "summary": {
                "pass_at_1": f"{acc_voted:.2f}%",
                "avg_nfe": avg_nfe,
                "avg_svf": avg_svf
            },
            "details": detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_report, out_f, ensure_ascii=False, indent=4)
        print(f"成功保存结果至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--res_path", type=str, default=RES_PATH)
    args = parser.parse_args()
    run_evaluation(args.res_path)