import torch
import torch.nn.functional as F
import numpy as np
from .verifier import CodeVerifier
import logging
import re
import math

logger = logging.getLogger(__name__)

class HTSSampler:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verifier = CodeVerifier(model, tokenizer, device)

    def _get_num_transfer_tokens(self, block_length, steps):
        if steps == 0: return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    def _sample_with_temperature(self, logits, temperature, top_k, top_p):
        logits = logits.to(torch.float32)
        
        orig_probs = torch.softmax(logits, dim=-1)
        x0_p, _ = torch.max(orig_probs, dim=-1)

        if temperature > 0.0:
            noise = torch.rand_like(logits, dtype=torch.float32)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
            logits = logits / temperature + gumbel_noise

        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        x0 = torch.argmax(logits, dim=-1)
        
        return x0, x0_p

    def _safe_scalar(self, val):
        if isinstance(val, torch.Tensor):
            if val.numel() > 1: return val.mean().item()
            return val.item()
        return float(val)

    def _analyze_structure(self, text, task_type="code"):
        score = 0.0
        stripped = text.strip()
        if task_type == "code":
            if len(stripped) < 5: return -0.1
            keywords = ["return", "print", "yield", "lambda", "class ", "def "]
            if any(k in stripped for k in keywords): score += 0.05
            if ":" in stripped: score += 0.02
            if "    " in text: score += 0.03
        elif task_type == "math":
            if "\\boxed{" in stripped: score += 0.1
            if "The answer is" in stripped: score += 0.05
            if len(stripped) < 10: return -0.1
            if "Step" in stripped and stripped.count("Step") > 15: score -= 0.2
        return score

    def _chunked_forward(self, x, chunk_size=32, slice_start=None):
        total_batch = x.shape[0]
        logits_list = []
        for i in range(0, total_batch, chunk_size):
            end_idx = min(i + chunk_size, total_batch)
            sub_x = x[i:end_idx]
            with torch.no_grad():
                outputs = self.model(input_ids=sub_x)
                sub_logits = outputs.logits
                if slice_start is not None:
                    s_start = slice_start if slice_start >= 0 else sub_logits.shape[1] + slice_start
                    sub_logits = sub_logits[:, s_start:, :]
                logits_list.append(sub_logits.detach().clone())
        return torch.cat(logits_list, dim=0)

    def _branch_and_resample(self, x, conf_scores, survivor_indices, target_width, mask_id, 
                             prompt_length, resample_window=5, task_type="code"):
        num_survivors = len(survivor_indices)
        if num_survivors == 0: return x[:target_width].clone(), conf_scores[:target_width].clone()

        if task_type == "math": resample_window = 12 
        elif task_type == "reasoning": resample_window = 10 
        elif task_type == "code": resample_window = 6 
        
        base_repeat = target_width // num_survivors
        remainder = target_width % num_survivors
        new_x_list = []
        new_conf_list = []
        
        for i in range(num_survivors):
            count = base_repeat + (1 if i < remainder else 0)
            if count == 0: continue
            
            survivor_x = x[survivor_indices[i]]
            survivor_conf = conf_scores[survivor_indices[i]]
            
            new_x_list.append(survivor_x.unsqueeze(0))
            new_conf_list.append(survivor_conf.unsqueeze(0))
            
            if count > 1:
                gen_part = survivor_x[prompt_length:]
                gen_conf = survivor_conf[prompt_length:]
                non_mask_indices = (gen_part != mask_id).nonzero(as_tuple=True)[0]
                
                for _ in range(count - 1):
                    perturbed_x = survivor_x.clone()
                    perturbed_conf = survivor_conf.clone()
                    
                    if len(non_mask_indices) > 0:
                        pool_size = min(resample_window * 2, len(non_mask_indices))
                        current_token_confs = gen_conf[non_mask_indices]
                        
                        _, candidate_indices = torch.topk(current_token_confs, k=pool_size, largest=False)
                        
                        num_to_perturb = min(resample_window, pool_size)
                        rand_indices = torch.randperm(pool_size, device=self.device)[:num_to_perturb]
                        selected_sub_indices = candidate_indices[rand_indices]
                        
                        target_indices_in_x = prompt_length + non_mask_indices[selected_sub_indices]
                        perturbed_x[target_indices_in_x] = mask_id
                        perturbed_conf[target_indices_in_x] = 0.0
                        
                    new_x_list.append(perturbed_x.unsqueeze(0))
                    new_conf_list.append(perturbed_conf.unsqueeze(0))
                    
        return torch.cat(new_x_list, dim=0), torch.cat(new_conf_list, dim=0)

    @torch.no_grad()
    def generate_hts(self, prompt_text, input_ids, problem_data=None, 
                     initial_N=1, final_K=1, survivor_K=None,
                     prune_step_pct=0.0, reward_mode="confidence",
                     temperature=0.7, block_length=32, steps=64, gen_length=1024, 
                     top_p=0.95, top_k=None, minimal_topk=1, threshold=0.9, 
                     eos_id=156892, mask_id=156895,
                     hts_mode=False, hts_start_pct=0.1, hts_end_pct=0.6, decay_factor=1.5,
                     hts_survivor_k=4, task_type="code", until=None, pruning_interval=0): 
        
        input_ids = input_ids.to(self.device)
        if input_ids.shape[0] == 1: input_ids = input_ids.repeat(initial_N, 1)

        schedule_map = {}
        ts_start, tr_end = 0, 0
        if not hts_mode:
            final_K_list = [final_K] if not isinstance(final_K, list) else final_K
            prune_pct_list = [prune_step_pct] if not isinstance(prune_step_pct, list) else prune_step_pct
            survivor_K_list = final_K_list if survivor_K is None else ([survivor_K] if not isinstance(survivor_K, list) else survivor_K)
            if len(survivor_K_list) < len(final_K_list): survivor_K_list.extend(final_K_list[len(survivor_K_list):])
            for pct, width, parents in zip(prune_pct_list, final_K_list, survivor_K_list):
                if pct > 0:
                    s = int(steps * pct)
                    schedule_map[s] = (width, parents)
        else:
            final_K_list = [final_K] if not isinstance(final_K, int) else [final_K]
            ts_start, tr_end = int(steps * hts_start_pct), int(steps * hts_end_pct)

        steps = min(steps, gen_length // minimal_topk)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        
        x = torch.full((initial_N, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        
        conf_scores = torch.zeros((initial_N, total_length), dtype=torch.float32, device=self.device)
        conf_scores[:, :prompt_length] = 1.0
        
        prefill_blocks = prompt_length // block_length
        num_gen_blocks = max(1, num_blocks - prefill_blocks)
        current_bsz = initial_N
        
        next_allowed_pruning_step = ts_start if hts_mode else 0
        
        stats = {
            "initial_n": initial_N, "final_k": final_K_list[-1], 
            "pruning_history": [], "entropy_history": [], "nfe": 0.0,
            "svf_calls": 0, "final_scores": [], "total_steps": steps
        }

        for num_block in range(prefill_blocks, num_blocks):
            window_end = (num_block + 1) * block_length
            schedule = self._get_num_transfer_tokens(block_length, steps)
            
            for step in range(steps):
                cur_x = x[:current_bsz, :window_end]
                
                perform_pruning = False
                num_parents_to_select = 0
                
                if hts_mode and step >= next_allowed_pruning_step and step < tr_end:
                    target_width = max(final_K_list[-1], math.ceil(initial_N * (decay_factor ** -(step - ts_start))))
                    if current_bsz > target_width:
                        perform_pruning = True
                        num_parents_to_select = hts_survivor_k 
                elif not hts_mode and step in schedule_map:
                    target_width, num_parents_to_select = schedule_map[step]
                    if current_bsz > target_width: perform_pruning = True
                
                if perform_pruning:
                    stats["nfe"] += current_bsz
                    stats["svf_calls"] += current_bsz
                    
                    gen_logits = self._chunked_forward(cur_x, chunk_size=16, slice_start=prompt_length)
                    rough_ids = torch.argmax(gen_logits, dim=-1)
                    rough_codes_snippet = self.tokenizer.batch_decode(rough_ids, skip_special_tokens=True)
                    candidates = []
                    for i in range(current_bsz):
                        full_code = rough_codes_snippet[i]
                        s = self._safe_scalar(self.verifier.get_reward(prompt_text, full_code, mode=reward_mode, problem_data=problem_data, current_logits=gen_logits[i] if reward_mode != "svf" else None, task_type=task_type))
                        s += self._analyze_structure(full_code, task_type=task_type)
                        clean_content = full_code.strip().replace(" ", "").replace("\n", "")
                        candidates.append({'score': s, 'idx': i, 'key': hash(clean_content[:200] + clean_content[-200:])})
                    
                    stats["pruning_history"].append({"step": step, "scores": [c['score'] for c in candidates]})
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    
                    selected_indices, seen_keys = [], set()
                    for cand in candidates:
                        if len(selected_indices) >= num_parents_to_select: break
                        if cand['key'] not in seen_keys:
                            selected_indices.append(cand['idx']); seen_keys.add(cand['key'])
                    
                    if len(selected_indices) < num_parents_to_select:
                        for cand in candidates:
                            if len(selected_indices) >= num_parents_to_select: break
                            if cand['idx'] not in selected_indices: selected_indices.append(cand['idx'])
                    
                    top_indices = torch.tensor(selected_indices, device=self.device)
                    x, conf_scores = self._branch_and_resample(x, conf_scores, top_indices, target_width, mask_id, prompt_length, task_type=task_type)
                    
                    current_bsz = target_width
                    cur_x = x[:current_bsz, :window_end]
                    next_allowed_pruning_step = step + 1 + pruning_interval

                active_mask = cur_x[:, -block_length:] == mask_id
                if active_mask.sum() == 0: break
                
                stats["nfe"] += current_bsz
                
                active_logits = self._chunked_forward(cur_x, chunk_size=32, slice_start=-block_length)
                
                with torch.no_grad():
                    if len(stats["entropy_history"]) < 32:
                        probs_for_stats = torch.softmax(active_logits.float(), dim=-1)
                        entropy_per_branch = (-(probs_for_stats * torch.log(probs_for_stats + 1e-10)).sum(dim=-1).mean(dim=-1)).cpu().numpy().tolist()
                        stats["entropy_history"].append(entropy_per_branch)
                
                x0, x0_p = self._sample_with_temperature(active_logits, temperature, top_k, top_p)
                
                num_transfer = schedule[step].item()
                confidence = torch.where(active_mask, x0_p, -torch.inf)
                transfer_idx = torch.zeros_like(x0, dtype=torch.bool)
                
                for b in range(current_bsz):
                    k_transfer = min(num_transfer, active_mask[b].sum().item())
                    active_indices = torch.where(active_mask[b])[0]
                    if (confidence[b] > threshold).sum().item() >= k_transfer:
                        conf_indices = torch.where((confidence[b] > threshold) & active_mask[b])[0]; transfer_idx[b, conf_indices] = True
                    elif len(active_indices) > 0:
                        _, topk_indices = torch.topk(confidence[b][active_indices], k=min(k_transfer, len(active_indices))); transfer_idx[b, active_indices[topk_indices]] = True
                
                if transfer_idx.any(): 
                    cur_x[:, -block_length:][transfer_idx] = x0[transfer_idx]
                    conf_scores[:current_bsz, window_end-block_length:window_end][transfer_idx] = x0_p[transfer_idx]
                
                if task_type in ["math", "reasoning"]:
                    for b in range(current_bsz):
                        gen_span = cur_x[b, prompt_length:window_end]
                        text_snippet = self.tokenizer.decode(gen_span, skip_special_tokens=True)
                        should_stop = False
                        if task_type == "reasoning" and ("###" in text_snippet):
                            should_stop = True
                        if task_type == "math" and ("\\boxed{" in text_snippet and "}" in text_snippet.split("\\boxed{")[-1]):
                            should_stop = True
                        
                        if should_stop:
                            non_mask_indices = (gen_span != mask_id).nonzero(as_tuple=True)[0]
                            if len(non_mask_indices) > 0:
                                last_idx = non_mask_indices[-1].item()
                                if last_idx + 1 < len(gen_span):
                                    gen_span[last_idx + 1:] = eos_id
                                    cur_x[b, prompt_length:window_end] = gen_span
                                if window_end < total_length: 
                                    x[b, window_end:] = eos_id
                                    conf_scores[b, window_end:] = 1.0 

                for b in range(current_bsz):
                    gen_window = cur_x[b, prompt_length:window_end]
                    eos_indices = (gen_window == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        first_eos_idx = eos_indices[0].item()
                        if first_eos_idx + 1 < len(gen_window):
                            gen_window[first_eos_idx + 1:] = eos_id
                            cur_x[b, prompt_length:window_end] = gen_window

            x = x[:current_bsz]
            x[:, :window_end] = cur_x

        stats["nfe"] = int(round(stats["nfe"]))
        
        final_gen_tokens = x[:current_bsz, prompt_length:]
        final_codes = self.tokenizer.batch_decode(final_gen_tokens, skip_special_tokens=True)
        final_candidates = []
        
        stats["svf_calls"] += len(final_codes)

        for i in range(len(final_codes)):
            txt = final_codes[i]
            if until:
                for term in until:
                    if term in txt: txt = txt.split(term)[0]
            s = self._safe_scalar(self.verifier.get_reward(prompt_text, txt, mode=reward_mode, task_type=task_type))
            s += self._analyze_structure(txt, task_type)
            final_candidates.append({'resp': txt, 'score': s})
            
        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        stats["final_scores"] = [c['score'] for c in final_candidates]
        stats["all_trajectories"] = [{"rank": i+1, "resp": c['resp'], "score": c['score']} for i, c in enumerate(final_candidates)]
        
        return [c['resp'] for c in final_candidates], stats