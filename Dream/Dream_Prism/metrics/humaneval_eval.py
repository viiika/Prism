import os
import sys
import json
import ast
import re
import glob
import argparse
import textwrap
import evaluate as hf_evaluate
from collections import Counter

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

RES_PATH = "<PATH_TO_RESULTS_JSONL>"

def strict_dedent(text: str) -> str:
    lines = text.split('\n')
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    
    if not lines:
        return ""

    min_indent = None
    for line in lines:
        if line.strip(): 
            indent = len(line) - len(line.lstrip())
            if min_indent is None or indent < min_indent:
                min_indent = indent
    
    if min_indent is None:
        min_indent = 0

    dedented_lines = []
    for line in lines:
        if line.strip():
            if len(line) >= min_indent:
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append(line.lstrip())
        else:
            dedented_lines.append("")
            
    return "\n".join(dedented_lines)

def extract_python_code(text: str) -> str:
    if not text:
        return ""
    
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").replace("<|notification_end|>", "")

    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1)

    code_block_pattern = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
    match = code_block_pattern.search(text)
    
    if match:
        content = match.group(1)
    else:
        if "```" in text:
            content = text.split("```")[0]
        else:
            lines = text.split('\n')
            cleaned_lines = []
            stop_words = ["Explanation:", "Example:", "Test Case:", "Output:", "Here are the tests:"]
            for line in lines:
                if any(sw in line for sw in stop_words):
                    break
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)

    return strict_dedent(content)

def normalize_code(code: str) -> str:
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

def sanitize(prompt: str, completion: str, entry_point: str) -> str:
    if f"def {entry_point}" in completion:
        imports = [line for line in prompt.split("\n") if line.startswith("import ") or line.startswith("from ")]
        return "\n".join(imports) + "\n" + completion

    clean_body = strict_dedent(completion)
    if not clean_body:
        return prompt 

    indented_body = "\n".join(["    " + line if line.strip() else "" for line in clean_body.split('\n')])
    return prompt.strip() + "\n" + indented_body

def perform_majority_voting(trajectories, prompt, entry_point):
    candidate_stats = {}

    for item in trajectories:
        if isinstance(item, dict):
            raw_text = item.get("resp", "")
            score = item.get("score", 0.0)
        else:
            raw_text = str(item[0] if isinstance(item, list) else item)
            score = 0.0

        extracted_code = extract_python_code(raw_text)
        full_code = sanitize(prompt, extracted_code, entry_point)
        
        is_valid = False
        try:
            ast.parse(full_code)
            is_valid = True
        except:
            is_valid = False

        norm_key = normalize_code(full_code)
        if not norm_key: continue

        if norm_key not in candidate_stats:
            candidate_stats[norm_key] = {
                "count": 0,
                "max_score": -float("inf"),
                "code": full_code,
                "is_valid": is_valid
            }
        
        candidate_stats[norm_key]["count"] += 1
        candidate_stats[norm_key]["max_score"] = max(candidate_stats[norm_key]["max_score"], score)

    if not candidate_stats:
        return prompt 

    sorted_candidates = sorted(
        candidate_stats.values(),
        key=lambda x: (x["is_valid"], x["count"], x["max_score"]),
        reverse=True
    )
    
    return sorted_candidates[0]["code"]

def run_evaluation(target_path):
    if os.path.isdir(target_path):
        jsonl_files = glob.glob(os.path.join(target_path, "*.jsonl"))
    else:
        jsonl_files = [target_path]

    try:
        code_eval = hf_evaluate.load("code_eval")
    except Exception as e:
        print(f"Error loading code_eval: {e}")
        return

    for file_path in jsonl_files:
        print(f">>> 正在评测文件: {file_path}")
        
        all_voted_predictions = [] 
        all_references = []        
        detailed_logs = []
        
        nfe_sum = 0
        svf_sum = 0
        valid_samples = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                except:
                    continue

                doc = data.get("doc", {})
                task_id = doc.get("task_id", f"Task_{valid_samples}")
                prompt = doc.get("prompt", "")
                entry_point = doc.get("entry_point", "solution")
                test_code = doc.get("test", "") + f"\ncheck({entry_point})"
                
                nfe_sum += data.get("nfe", 0)
                svf_sum += data.get("svf_calls", 0)
                valid_samples += 1

                trajectories = data.get("all_trajectories", data.get("resps", []))
                voted_code = perform_majority_voting(trajectories, prompt, entry_point)

                all_voted_predictions.append([voted_code]) 
                all_references.append(test_code)

                detailed_logs.append({
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "final_code": voted_code,
                    "nfe": data.get("nfe", 0),
                    "svf": data.get("svf_calls", 0),
                })

        if not all_voted_predictions: continue

        print(f"执行测试中...")
        pass_at_k, exec_results = code_eval.compute(
            references=all_references,
            predictions=all_voted_predictions,
            k=[1]
        )

        accuracy = pass_at_k.get("pass@1", 0.0) * 100
        avg_nfe = nfe_sum / valid_samples if valid_samples > 0 else 0
        avg_svf = svf_sum / valid_samples if valid_samples > 0 else 0

        for i, log in enumerate(detailed_logs):
            res = exec_results.get(i, [])
            log["passed"] = res[0][1].get("passed", False) if res else False
            log["exec_msg"] = res[0][1].get("result", "failed") if res else "failed"

        output_path = file_path.replace(".jsonl", "_voted_result.json")
        final_report = {
            "meta": {"file": file_path, "total_samples": valid_samples},
            "metrics": {"accuracy": f"{accuracy:.2f}%", "avg_nfe": avg_nfe, "avg_svf": avg_svf},
            "details": detailed_logs
        }

        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_report, out_f, ensure_ascii=False, indent=4)
        print(f"Accuracy: {accuracy:.2f}% | SVF: {avg_svf:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--res_path", type=str, default=RES_PATH)
    args = parser.parse_args()
    run_evaluation(args.res_path)