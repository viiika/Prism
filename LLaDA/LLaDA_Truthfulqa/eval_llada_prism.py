'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
And extended with Prism methods for LLaDA-Instruct.
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import json
import os
import logging
import math
import textwrap
import time 
from collections import Counter

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

class CodeVerifier:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.yes_ids, self.no_ids = [], []
        for t in ["Yes", " Yes", "YES"]:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if ids: self.yes_ids.append(ids[-1])
        for t in ["No", " No", "NO"]:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if ids: self.no_ids.append(ids[-1])
        self.yes_ids = list(set(self.yes_ids))
        self.no_ids = list(set(self.no_ids))

    def svf_score(self, prompt, code_str, task_type="code"):
        max_len = 2000
        truncated_code = code_str[:max_len]
        
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

        **Conclusion**: Based on your analysis, is the answer fully faithful to the context and correct? Answer with a single word: Yes or No.
        **Answer:** """
        elif task_type == "qa":
            prompt_template = f"""
        You are a strict truthfulness judge. Your task is to verify if the answer is factually accurate and rejects common human misconceptions or popular urban legends.

        [Question]
        {prompt}
        [/Question]

        [Proposed Answer]
        {truncated_code}
        [/Proposed Answer]

        **Conclusion**: Based on these criteria, is this answer fully truthful and accurate? Answer with a single word: Yes or No.
        **Answer:** """
        else:
            prompt_template = f"Is the following answer correct?\nQuestion: {prompt}\nAnswer: {truncated_code}\nAnswer Yes or No.\nAnswer:"

        verify_text = textwrap.dedent(prompt_template).strip()
        input_ids = self.tokenizer(verify_text, return_tensors="pt").input_ids.to(self.device)
        if input_ids.shape[1] > 2048: input_ids = input_ids[:, -2048:]

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            yes_score = max([logits[i].item() for i in self.yes_ids if i < logits.shape[-1]] + [-100.0])
            no_score = max([logits[i].item() for i in self.no_ids if i < logits.shape[-1]] + [-100.0])
            probs = torch.softmax(torch.tensor([yes_score, no_score]), dim=0)
            return probs[0].item()

    def get_reward(self, prompt, code_str, mode="confidence", current_logits=None, task_type="code"):
        if mode == "svf":
            return self.svf_score(prompt, code_str, task_type=task_type)
        else:
            if current_logits is None: return 0.0
            probs = torch.softmax(current_logits.to(torch.float32), dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            return torch.exp(torch.mean(torch.log(max_probs + 1e-10))).item()

class HTSSampler:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verifier = CodeVerifier(model, tokenizer, device)

    def _sample_with_temperature(self, logits, temperature=0.7):
        logits = logits.to(torch.float32)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(logits.shape[:-1])
            x0_p = torch.gather(torch.softmax(logits, dim=-1), -1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            x0_p, x0 = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        return x0, x0_p

    @torch.no_grad()
    def generate_hts(self, prompt_text, input_ids, initial_N=1, final_K=1, hts_survivor_k=2,
                     steps=32, gen_length=32, mask_id=126336, reward_mode="svf", task_type="qa", 
                     decay_factor=1.8, hts_start_pct=0.1, hts_end_pct=0.6, pruning_interval=3):
        
        b = initial_N
        prompt_len = input_ids.shape[1]
        xt = torch.full((b, prompt_len + gen_length), mask_id, dtype=torch.long, device=self.device)
        xt[:, :prompt_len] = input_ids.repeat(b, 1)
        
        conf_scores = torch.zeros((b, prompt_len + gen_length), device=self.device)
        ts_start, tr_end = int(steps * hts_start_pct), int(steps * hts_end_pct)
        
        schedule = torch.full((steps,), gen_length // steps, dtype=torch.int64, device=self.device)
        schedule[:gen_length % steps] += 1
        
        next_pruning = ts_start
        for i in range(steps):
            mask_indices = (xt == mask_id)
            if not mask_indices.any(): break
            
            logits = self.model(xt).logits
            x0, x0_p = self._sample_with_temperature(logits[:, prompt_len:], temperature=0.7)
            
            # Update tokens based on confidence
            for idx in range(b):
                curr_mask = mask_indices[idx, prompt_len:]
                if not curr_mask.any(): continue
                conf = torch.where(curr_mask, x0_p[idx], -float('inf'))
                _, sel_idx = torch.topk(conf, k=min(schedule[i].item(), curr_mask.sum().item()))
                xt[idx, prompt_len + sel_idx] = x0[idx, sel_idx]
                conf_scores[idx, prompt_len + sel_idx] = x0_p[idx, sel_idx]

            # Pruning
            if i >= next_pruning and i < tr_end and b > final_K:
                target_width = max(final_K, math.ceil(initial_N * (decay_factor ** -(i - ts_start))))
                if b > target_width:
                    scores = []
                    decoded_texts = self.tokenizer.batch_decode(xt[:, prompt_len:], skip_special_tokens=True)
                    for j in range(b):
                        s = self.verifier.get_reward(prompt_text, decoded_texts[j], mode=reward_mode, 
                                                   task_type=task_type, current_logits=logits[j, prompt_len:])
                        scores.append(s)
                    
                    top_k_indices = torch.topk(torch.tensor(scores), k=min(target_width, b)).indices
                    xt = xt[top_k_indices]
                    conf_scores = conf_scores[top_k_indices]
                    b = xt.shape[0]
                    next_pruning = i + pruning_interval

        # Final decoding and ranking
        final_texts = self.tokenizer.batch_decode(xt[:, prompt_len:], skip_special_tokens=True)
        results = []
        for j in range(b):
            s = self.verifier.get_reward(prompt_text, final_texts[j], mode=reward_mode, task_type=task_type)
            results.append({'text': final_texts[j], 'score': s})
        
        results.sort(key=lambda v: v['score'], reverse=True)
        return [r['text'] for r in results]

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(self, model_path='', mask_id=126336, max_length=4096, generated_samples_path='',
                 batch_size=32, sampling_steps=64, mask_length=128, sampler='hts', task_type="qa",
                 hts_initial_n=8, final_K=1, hts_reward_mode="svf", hts_start_pct=0.1, hts_end_pct=0.6,
                 **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = self.model.device

        self.mask_id = mask_id
        self.sampling_steps = int(sampling_steps)
        self.mask_length = int(mask_length)
        self.sampler = sampler
        self.task_type = task_type
        self.generated_samples_path = generated_samples_path
        
        self.hts_initial_n = int(hts_initial_n)
        self.final_K = int(final_K)
        self.hts_reward_mode = hts_reward_mode
        self.hts_start_pct = float(hts_start_pct)
        self.hts_end_pct = float(hts_end_pct)
        
        self.hts_sampler = HTSSampler(self.model, self.tokenizer, self.device)
        self._rank = 0

    @torch.no_grad()
    def llada_conf_sample(self, prompt):
        xt = torch.full((1, prompt.shape[1] + self.mask_length), self.mask_id, dtype=torch.long, device=self.device)
        xt[:, :prompt.shape[1]] = prompt
        
        step_size = self.mask_length // self.sampling_steps
        for i in range(self.sampling_steps):
            mask_indices = (xt == self.mask_id)
            if not mask_indices.any(): break
            logits = self.model(xt).logits
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = _sample_categorical(p)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            confidence = torch.where(mask_indices, x0_p, -float('inf'))
            _, select_idx = torch.topk(confidence[0], k=min(step_size, mask_indices.sum().item()))
            xt[0, select_idx] = x0[0, select_idx]
        return xt

    @torch.no_grad()
    def generate_until(self, requests):
        start_time = time.time()
        
        out, out_for_json = [], []
        for req in tqdm(requests, desc="Generating..."):
            prompt_text = req.args[0]
            until = req.args[1]['until']
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)

            if self.sampler == 'hts':
                candidates = self.hts_sampler.generate_hts(
                    prompt_text=prompt_text,
                    input_ids=prompt_ids,
                    initial_N=self.hts_initial_n,
                    final_K=self.final_K,
                    steps=self.sampling_steps,
                    gen_length=self.mask_length,
                    reward_mode=self.hts_reward_mode,
                    task_type=self.task_type,
                    hts_start_pct=self.hts_start_pct,
                    hts_end_pct=self.hts_end_pct
                )
                if not candidates:
                    generated_answer = ""
                else:
                    counts = Counter(candidates)
                    most_common = counts.most_common()
                    if most_common[0][1] > 1:
                        generated_answer = most_common[0][0]
                    else:
                        generated_answer = candidates[0]
            else:
                res_ids = self.llada_conf_sample(prompt_ids)
                generated_answer = self.tokenizer.decode(res_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)

            for stop_seq in until + ["<|eot_id|>", self.tokenizer.eos_token]:
                if stop_seq and stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer = generated_answer.strip()
            out.append(generated_answer)
            out_for_json.append({"prefix": prompt_text, "result": generated_answer})


        end_time = time.time()
        total_duration = end_time - start_time

        if self.generated_samples_path:
            os.makedirs(self.generated_samples_path, exist_ok=True)
            final_output = {
                "total_time_seconds": total_duration,
                "samples": out_for_json
            }
            with open(os.path.join(self.generated_samples_path, "res.json"), "w") as f:
                json.dump(final_output, f, indent=2)
        return out

    def loglikelihood(self, requests): return []
    def loglikelihood_rolling(self, requests): return []
    @property
    def rank(self): return 0
    @property
    def world_size(self): return 1

if __name__ == "__main__":
    cli_evaluate()