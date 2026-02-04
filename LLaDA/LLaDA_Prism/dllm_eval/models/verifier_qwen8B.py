import torch
import logging
import ast
import re
import numpy as np
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class CodeVerifier:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        reward_model_path = "PATH_TO_YOUR_Qwen3-8B"
        logger.info(f"Loading Qwen3-8B as reward model from {reward_model_path}...")
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path, 
            trust_remote_code=True
        )
        
        if self.reward_tokenizer.pad_token_id is None:
            self.reward_tokenizer.pad_token_id = 151643

        self.reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model_path,
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        self.yes_ids, self.no_ids = [], []
        for t in ["Yes", " Yes", "YES"]:
            ids = self.reward_tokenizer.encode(t, add_special_tokens=False)
            if len(ids) > 0: self.yes_ids.append(ids[-1])
        for t in ["No", " No", "NO"]:
            ids = self.reward_tokenizer.encode(t, add_special_tokens=False)
            if len(ids) > 0: self.no_ids.append(ids[-1])
            
        self.yes_ids = list(set(self.yes_ids))
        self.no_ids = list(set(self.no_ids))

    def _extract_python_code(self, text):
        text = text.strip()
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match: return match.group(1)
        match_generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match_generic: return match_generic.group(1)
        return text

    def check_syntax(self, code_str):
        clean_code = self._extract_python_code(code_str)
        try:
            if len(clean_code.strip()) < 5: return False
            ast.parse(clean_code)
            return True
        except:
            return False

    def compute_confidence(self, logits):
        if logits is None: return 0.0
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        log_probs = torch.log(max_probs + 1e-10)
        return torch.exp(torch.mean(log_probs)).item()

    def svf_score(self, prompt, code_str, task_type="code"):
        max_char_len = 4000 
        if len(code_str) > max_char_len:
            if task_type == "reasoning":
                truncated_code = code_str[:1000] + "\n...[truncated]...\n" + code_str[-(max_char_len-1000):]
            else:
                truncated_code = code_str[-max_char_len:]
        else:
            truncated_code = code_str
        
        if task_type == "code":
            prompt_template = f"""
        You are an expert programming contest judge. Your task is to evaluate a generated solution for a given problem based on correctness, efficiency, and adherence to constraints.

        [Problem Statement]
        {prompt}
        [/Problem Statement]

        [Proposed Python Solution]
        ```python
        {truncated_code}
        ```
        [/Proposed Python Solution]

        **Analysis Steps:**
        1. Correctness: Does the core algorithm correctly solve the problem?
        2. Efficiency: Is the time complexity acceptable for the given constraints?
        3. Edge Cases & Constraints: Does the code handle all rules and edge cases?

        **Conclusion**: Based on your analysis, is the solution likely to be fully correct? Answer with a single word: Yes or No.
        **Answer:** """
        elif task_type == "math":
            prompt_template = f"""
        You are an expert mathematician and competition judge. Your task is to evaluate a proposed mathematical solution for a given problem based on its logical rigor and accuracy.

        [Math Problem]
        {prompt}
        [/Math Problem]

        [Proposed Mathematical Solution]
        {truncated_code}
        [/Proposed Mathematical Solution]

        **Analysis Steps:**
        1. Reasoning Validity: Are the logical steps and mathematical properties applied correctly?
        2. Calculation Accuracy: Are the intermediate calculations or algebraic manipulations accurate?
        3. Goal Alignment: Does the current reasoning path directly lead toward the final answer required by the problem?

        **Conclusion**: Based on your analysis, is this solution path sound and likely to result in the correct final answer? Answer with a single word: Yes or No.
        **Answer:** """
        elif task_type == "reasoning":
            prompt_template = f"""
        You are an expert reading comprehension and faithfulness judge. Your task is to evaluate a generated answer based on the provided context and question.

        [Context and Question]
        {prompt}
        [/Context and Question]

        [Proposed Answer]
        {truncated_code}
        [/Proposed Answer]

        **Analysis Steps :**
        1. Faithfulness: Is the answer an exact, literal span from the context?
        2. Relevance: Does the answer directly address the specific question asked without hallucinating external information?
        3. Accuracy: Does the provided context strictly support this answer? 

        **Conclusion**: Based on your analysis, is the answer fully faithful to the context and correct? Answer with a single word: Yes or No.
        **Answer:** """
        else:
            prompt_template = f"Is the following answer correct?\nQuestion: {prompt}\nAnswer: {truncated_code}\nAnswer Yes or No.\nAnswer:"

        verify_text = textwrap.dedent(prompt_template).strip()
        
        input_ids = self.reward_tokenizer(verify_text, return_tensors="pt").input_ids.to(self.reward_model.device)
    
        max_pos = getattr(self.reward_model.config, "max_position_embeddings", 40960)
        if input_ids.shape[1] > max_pos - 10:
            input_ids = input_ids[:, -(max_pos - 10):]

        with torch.no_grad():
            outputs = self.reward_model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            yes_score = max((logits[i].item() for i in self.yes_ids if i < logits.shape[-1]), default=-float('inf'))
            no_score = max((logits[i].item() for i in self.no_ids if i < logits.shape[-1]), default=-float('inf'))
            
            if yes_score == -float('inf') and no_score == -float('inf'): return 0.5
            
            probs = torch.softmax(torch.tensor([yes_score, no_score]), dim=0)
            return probs[0].item()

    def get_reward(self, prompt, code_str, mode="confidence", problem_data=None, current_logits=None, task_type="code"):
        if mode == "svf":
            return self.svf_score(prompt, code_str, task_type=task_type)
        else:
            return self.compute_confidence(current_logits)