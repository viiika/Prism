import os
import sys
import json
import ast
import traceback
import glob
import math
import argparse
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import evaluate as hf_evaluate
import re

RES_PATH = "<PATH_TO_RESULTS_JSONL>"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def extract_python_code(text: str) -> str:
    if not text: return ""
    
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").replace("<|notification_end|>", "")
    
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1)
    
    if "```python" in text:
        content = text.split("```python")[-1]
        if "```" in content:
            return content.split("```")[0].strip()
        return content.strip()
    elif "```" in text:
        content = text.split("```")[-1]
        if "```" in content:
            return content.split("```")[0].strip()
        return content.strip()

    lines = text.split('\n')
    cleaned_lines = []
    stop_words = ["Explanation:", "Example:", "Test Case:", "Output:"]
    for line in lines:
        if any(sw in line for sw in stop_words):
            break
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

def sanitize(prompt: str, completion: str, entrypoint: str) -> str:
    if f"def {entrypoint}" in completion:
        return completion
    return prompt + "\n" + completion

def run_evaluation(target_path):
    if os.path.isdir(target_path):
        jsonl_files = glob.glob(os.path.join(target_path, "**/*.jsonl"), recursive=True)
    else:
        jsonl_files = [target_path]

    if not jsonl_files:
        print(f"未在路径 {target_path} 下找到任何 .jsonl 文件")
        return

    print(f"共找到 {len(jsonl_files)} 个评测任务")
    code_eval = hf_evaluate.load("code_eval")

    for file_path in jsonl_files:
        print(f"\n>>> 正在评测: {file_path}")
        all_predictions = []
        all_references = []
        detailed_results = []
        nfe_list = []
        svf_list = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines: continue
            
            for line in lines:
                if not line.strip(): continue
                item = json.loads(line)
                doc = item.get("doc", {})
                prompt = doc.get("prompt", "")
                entry_point = doc.get("entry_point", "")
                reference = doc.get("test", "") 
                
                current_nfe = item.get("nfe", 0)
                nfe_list.append(current_nfe)
                svf_list.append(item.get("svf_calls", 0))

                resps = item.get("resps", [])
                candidate_stats = {}
                
                for r in resps:
                    raw_text = r[0] if isinstance(r, list) else r
                    completion = extract_python_code(raw_text)
                    full_code = sanitize(prompt, completion, entry_point)
                    
                    try:
                        ast.parse(full_code)
                        is_valid = True
                    except:
                        is_valid = False
                        
                    logic_norm = normalize_code_for_voting(full_code)
                    if not logic_norm: continue
                    
                    if logic_norm not in candidate_stats:
                        candidate_stats[logic_norm] = {"count": 0, "valid": is_valid, "code": full_code}
                    candidate_stats[logic_norm]["count"] += 1

                if not candidate_stats:
                    voted_code = prompt
                else:
                    sorted_logics = sorted(
                        candidate_stats.keys(),
                        key=lambda k: (candidate_stats[k]["valid"], candidate_stats[k]["count"]),
                        reverse=True
                    )
                    voted_code = candidate_stats[sorted_logics[0]]["code"]

                all_predictions.append([voted_code])
                all_references.append(reference)
                detailed_results.append({
                    "task_id": doc.get("task_id", doc.get("name", "N/A")),
                    "voted_code": voted_code,
                    "nfe": current_nfe,
                    "svf_calls": item.get("svf_calls", 0),
                    "candidates_count": len(candidate_stats)
                })

        if not all_predictions: continue

        print(f"正在执行代码测试 (共 {len(all_predictions)} 题)...")
        pass_at_k, exec_results = code_eval.compute(
            references=all_references,
            predictions=all_predictions,
            k=[1],
            num_workers=4
        )

        accuracy = pass_at_k.get("pass@1", 0.0) * 100
        avg_nfe = int(round(sum(nfe_list) / len(nfe_list))) if nfe_list else 0
        avg_svf = int(round(sum(svf_list) / len(svf_list))) if svf_list else 0

        print(f"Accuracy: {accuracy:.2f}% | NFE: {avg_nfe} | SVF: {avg_svf} ---")

        output_name = f"eval_voted_{os.path.basename(file_path).replace('.jsonl', '.json')}"
        output_path = os.path.join(os.path.dirname(file_path), output_name)
        
        for i, detail in enumerate(detailed_results):
            res_list = exec_results.get(i, [])
            detail["is_correct"] = res_list[0][1]["passed"] if res_list else False

        final_report = {
            "summary": {
                "accuracy": f"{accuracy:.2f}%",
                "nfe": avg_nfe,
                "svf_calls": avg_svf
            },
            "details": detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_report, out_f, ensure_ascii=False, indent=4)
        print(f"报告已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--res_path", type=str, default=RES_PATH)
    args = parser.parse_args()
    run_evaluation(args.res_path)