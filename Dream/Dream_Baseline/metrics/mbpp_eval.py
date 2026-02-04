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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    text = text.replace("[DONE]", "") 

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
            start_idx = 0
            stop_words = ["Here is", "Explanation", "Example", "Note", "python", "The code"]
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("def ", "import ", "from ", "class ")):
                    start_idx = i
                    break
                if any(sw in line for sw in stop_words) and not stripped.endswith(":"):
                    continue
            
            content = "\n".join(lines[start_idx:])

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

def perform_majority_voting(trajectories):
    candidate_stats = {}

    for item in trajectories:
        if isinstance(item, dict):
            raw_text = item.get("resp", "")
            score = item.get("score", 0.0)
        elif isinstance(item, (list, tuple)):
            raw_text = item[0]
            score = 0.0
        else:
            raw_text = str(item)
            score = 0.0

        extracted_code = extract_python_code(raw_text)

        if not extracted_code.strip():
            continue
            
        is_valid = False
        try:
            ast.parse(extracted_code)
            is_valid = True
        except:
            is_valid = False

        norm_key = normalize_code(extracted_code)
        if not norm_key: continue

        if norm_key not in candidate_stats:
            candidate_stats[norm_key] = {
                "count": 0,
                "max_score": -float("inf"),
                "code": extracted_code,
                "is_valid": is_valid
            }
        
        candidate_stats[norm_key]["count"] += 1
        candidate_stats[norm_key]["max_score"] = max(candidate_stats[norm_key]["max_score"], score)

    if not candidate_stats:
        return ""

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
        print(f"\n>>> 正在评测 MBPP 文件: {file_path}")
        
        all_voted_predictions = [] 
        all_references = []        
        detailed_logs = []
        
        nfe_total = 0
        svf_total = 0
        count_valid_samples = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc = data.get("doc", {})
            task_id = doc.get("task_id", f"MBPP_{idx}")
            
            test_list = doc.get("test_list", [])
            test_setup = doc.get("test_setup_code", "")
            challenge_tests = doc.get("challenge_test_list", [])

            full_test_code = ""
            if test_setup:
                full_test_code += test_setup + "\n"
            if test_list:
                full_test_code += "\n".join(test_list) + "\n"
            if challenge_tests:
                full_test_code += "\n".join(challenge_tests)
            
            current_nfe = data.get("nfe", 0)
            current_svf = data.get("svf_calls", 0)
            
            nfe_total += current_nfe
            svf_total += current_svf
            count_valid_samples += 1

            trajectories = data.get("all_trajectories", [])
            if not trajectories:
                resps = data.get("resps", [])
                trajectories = [{"resp": r} for r in resps]

            voted_code = perform_majority_voting(trajectories)

            if not voted_code:
                voted_code = "def placeholder(): pass"

            all_voted_predictions.append([voted_code]) 
            all_references.append(full_test_code)

            detailed_logs.append({
                "task_id": task_id,
                "final_code": voted_code,
                "reference": full_test_code, 
                "nfe": current_nfe,
                "svf": current_svf,
                "traj_count": len(trajectories)
            })

        if not all_voted_predictions:
            print("未找到有效数据。")
            continue

        print(f"正在执行代码测试 (共 {len(all_voted_predictions)} 题)...")
        
        pass_at_k, exec_results = code_eval.compute(
            references=all_references,
            predictions=all_voted_predictions,
            k=[1],
            num_workers=4 
        )

        accuracy = pass_at_k.get("pass@1", 0.0) * 100
        avg_nfe = nfe_total / count_valid_samples if count_valid_samples > 0 else 0
        avg_svf = svf_total / count_valid_samples if count_valid_samples > 0 else 0
        print(f"Accuracy: {accuracy:.2f}% | NFE: {avg_nfe:.1f} | SVF: {avg_svf:.1f}")

        for i, log in enumerate(detailed_logs):
            res = exec_results.get(i, [])
            if res and len(res) > 0:
                is_passed = res[0][1].get("passed", False)
                eval_result_str = res[0][1].get("result", "passed") if not is_passed else "passed"
            else:
                is_passed = False
                eval_result_str = "Execution Failed"
            
            log["passed"] = is_passed
            log["exec_msg"] = eval_result_str

        output_name = f"eval_mbpp_{os.path.basename(file_path).replace('.jsonl', '.json')}"
        output_path = os.path.join(os.path.dirname(file_path), output_name)
        
        final_report = {
            "meta": {
                "file": file_path,
                "total_samples": count_valid_samples
            },
            "metrics": {
                "accuracy": f"{accuracy:.2f}%",
                "avg_nfe": avg_nfe,
                "avg_svf": avg_svf
            },
            "details": detailed_logs
        }

        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(final_report, out_f, ensure_ascii=False, indent=4)
        print(f"结果已保存至: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBPP Metrics Evaluation Script")
    parser.add_argument("-r", "--res_path", type=str, default=RES_PATH, help="Path to jsonl result file or directory")
    args = parser.parse_args()
    
    if os.path.exists(args.res_path):
        run_evaluation(args.res_path)
    else:
        print(f"Path not found: {args.res_path}")