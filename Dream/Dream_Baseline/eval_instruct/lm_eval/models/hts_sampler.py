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
        return score

    def _chunked_forward(self, x, chunk_size=32, slice_indices=None):
        total_batch = x.shape[0]
        logits_list = []
        for i in range(0, total_batch, chunk_size):
            end_idx = min(i + chunk_size, total_batch)
            sub_x = x[i:end_idx]
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(sub_x, 'full')
                    sub_logits = outputs.logits
                    sub_logits = torch.cat([sub_logits[:, :1, :], sub_logits[:, :-1, :]], dim=1)
                if slice_indices is not None:
                    s_start, s_end = slice_indices
                    sub_logits = sub_logits[:, s_start:s_end, :]
                logits_list.append(sub_logits.detach().clone())
        return torch.cat(logits_list, dim=0)

    def _branch_and_resample(self, x, conf_scores, survivor_indices, target_width, mask_id, 
                             prompt_length, resample_window=6, task_type="code"):
        num_survivors = len(survivor_indices)
        if num_survivors == 0: return x[:target_width].clone(), conf_scores[:target_width].clone()

        base_repeat = target_width // num_survivors
        remainder = target_width % num_survivors
        new_x_list, new_conf_list = [], []
        
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
                        _, candidate_pool = torch.topk(current_token_confs, k=pool_size, largest=False)
                        
                        num_to_perturb = min(resample_window, pool_size)
                        rand_indices = torch.randperm(pool_size, device=self.device)[:num_to_perturb]
                        selected_sub_indices = candidate_pool[rand_indices]
                        
                        target_idx_in_x = prompt_length + non_mask_indices[selected_sub_indices]
                        perturbed_x[target_idx_in_x] = mask_id
                        perturbed_conf[target_idx_in_x] = 0.0
                    new_x_list.append(perturbed_x.unsqueeze(0))
                    new_conf_list.append(perturbed_conf.unsqueeze(0))
                    
        return torch.cat(new_x_list, dim=0), torch.cat(new_conf_list, dim=0)

    @torch.no_grad()
    def generate_hts(self, prompt_text, input_ids, problem_data=None, 
                     initial_N=1, final_K=1, survivor_K=None,
                     prune_step_pct=0.0, reward_mode="confidence",
                     temperature=0.7, block_length=32, steps=64, gen_length=1024, 
                     top_p=0.95, top_k=None, minimal_topk=1, threshold=0.9, 
                     eos_id=151643, mask_id=151666,
                     hts_mode=False, hts_start_pct=0.1, hts_end_pct=0.6, decay_factor=1.2,
                     hts_survivor_k=4, task_type="code", until=None, pruning_interval=20): 
        
        input_ids = input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        total_length = prompt_length + gen_length
        
        x = torch.full((initial_N, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.repeat(initial_N, 1)
        conf_scores = torch.zeros((initial_N, total_length), dtype=torch.float32, device=self.device)
        conf_scores[:, :prompt_length] = 1.0
        
        schedule = self._get_num_transfer_tokens(gen_length, steps)
        current_bsz = initial_N
        schedule_map = {}
        ts_start, tr_end = 0, 0
        
        if hts_mode:
            ts_start, tr_end = int(steps * hts_start_pct), int(steps * hts_end_pct)
        else:
            final_K_list = [final_K] if not isinstance(final_K, list) else final_K
            prune_pct_list = [prune_step_pct] if not isinstance(prune_step_pct, list) else prune_step_pct
            for pct, width in zip(prune_pct_list, final_K_list):
                if pct > 0: schedule_map[int(steps * pct)] = width

        stats = {
            "initial_n": initial_N,
            "final_k": final_K if not isinstance(final_K, list) else final_K[-1],
            "nfe": 0,
            "svf_calls": 0,
            "pruning_history": [],
            "entropy_history": [],
            "final_scores": []
        }
        
        next_allowed_pruning_step = ts_start

        for step in range(steps):
            perform_pruning = False
            num_parents_to_select = hts_survivor_k
            
            if hts_mode and ts_start <= step < tr_end and step >= next_allowed_pruning_step:
                target_width = max(stats["final_k"], math.ceil(initial_N * (decay_factor ** -(step - ts_start))))
                if current_bsz > target_width: 
                    perform_pruning = True
            elif not hts_mode and step in schedule_map:
                target_width = schedule_map[step]
                num_parents_to_select = target_width
                if current_bsz > target_width: 
                    perform_pruning = True

            if perform_pruning:
                stats["svf_calls"] += current_bsz
                full_logits = self._chunked_forward(x[:current_bsz, :], slice_indices=(prompt_length, total_length))
                rough_ids = torch.argmax(full_logits, dim=-1)
                rough_codes = self.tokenizer.batch_decode(rough_ids, skip_special_tokens=True)
                
                candidates = []
                for i in range(current_bsz):
                    s = self._safe_scalar(self.verifier.get_reward(prompt_text, rough_codes[i], mode=reward_mode, current_logits=full_logits[i], task_type=task_type))
                    s += self._analyze_structure(rough_codes[i], task_type=task_type)
                    clean_text = rough_codes[i].strip().replace(" ", "").replace("\n", "")
                    content_key = hash(clean_text[:150] + clean_text[-150:]) if clean_text else i
                    candidates.append({'score': s, 'idx': i, 'key': content_key})
                
                stats["pruning_history"].append({"step": step, "scores": [c['score'] for c in candidates]})
                candidates.sort(key=lambda c: c['score'], reverse=True)
                
                selected_indices, seen_keys = [], set()
                for cand in candidates:
                    if len(selected_indices) >= num_parents_to_select: break
                    if cand['key'] not in seen_keys:
                        selected_indices.append(cand['idx']); seen_keys.add(cand['key'])
                for cand in candidates:
                    if len(selected_indices) >= num_parents_to_select: break
                    if cand['idx'] not in selected_indices: selected_indices.append(cand['idx'])
                
                top_indices = torch.tensor(selected_indices, device=self.device)
                x, conf_scores = self._branch_and_resample(x, conf_scores, top_indices, target_width, mask_id, prompt_length, task_type=task_type)
                
                current_bsz = target_width
                next_allowed_pruning_step = step + pruning_interval

            active_mask = (x[:current_bsz, prompt_length:] == mask_id)
            
            stats["nfe"] += current_bsz
            logits = self._chunked_forward(x[:current_bsz, :], slice_indices=(prompt_length, total_length))
            
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                stats["entropy_history"].append(entropy)
            
            x0, x0_p = self._sample_with_temperature(logits, temperature, top_k, top_p)
            num_transfer = schedule[step].item()
            
            confidence = torch.where(active_mask, x0_p, -torch.inf)
            transfer_idx = torch.zeros_like(x0, dtype=torch.bool)
            
            for b in range(current_bsz):
                k = min(num_transfer, active_mask[b].sum().item())
                if k <= 0: continue 
                high_conf_mask = (confidence[b] > threshold)
                if high_conf_mask.sum() >= k:
                    transfer_idx[b] = high_conf_mask
                else:
                    _, topk_ids = torch.topk(confidence[b], k=k)
                    transfer_idx[b, topk_ids] = True
            
            if transfer_idx.any():
                x[:current_bsz, prompt_length:][transfer_idx] = x0[transfer_idx]
                conf_scores[:current_bsz, prompt_length:][transfer_idx] = x0_p[transfer_idx]

        final_codes = self.tokenizer.batch_decode(x[:current_bsz, prompt_length:], skip_special_tokens=True)
        final_candidates = []
        for i, code in enumerate(final_codes):
            txt = code.split(self.tokenizer.eos_token)[0]
            if until:
                for term in until:
                    if term in txt: txt = txt.split(term)[0]
            s = self._safe_scalar(self.verifier.get_reward(prompt_text, txt, mode=reward_mode, task_type=task_type))
            final_candidates.append({'resp': txt, 'score': s})
            
        final_candidates.sort(key=lambda c: c['score'], reverse=True)
        stats["final_scores"] = [c['score'] for c in final_candidates]
        stats["all_trajectories"] = [{"rank": i+1, "resp": c['resp'], "score": c['score']} for i, c in enumerate(final_candidates)]
        
        return [c['resp'] for c in final_candidates], stats