import logging
import gc
import random
import json
import os
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, get_dtype

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def empty_cache_by_memory(threshold_gb=70):
    """
    Empty CUDA cache if allocated memory exceeds threshold
    Args:
        threshold_gb: Memory threshold in GB
    """
    if torch.cuda.is_available():
        # Get current memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB

        if allocated > threshold_gb:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")

@register_model("diffllm")
class DiffLLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        max_new_tokens: Optional[int] = 128,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        classifier_free_guidance: Optional[float] = 1.0,
        pad_to_max_len: Optional[bool] = False,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        
        self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            self._device = self.accelerator.device

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or (hasattr(self, "accelerator") and self.accelerator.num_processes > 1)):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if self.accelerator.num_processes > 1:
                    self._device = torch.device(f"{accelerator.device}")
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
            else:
                self._rank = 0
                self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.alg = kwargs.get("alg", "entropy")
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        self.top_k = kwargs.get("top_k", None)

        self.nll_type = nll_type
        self.log_type = log_type
        self.classifier_free_guidance = classifier_free_guidance
        self.pad_to_max_len = pad_to_max_len
        self.sampling_eps = sampling_eps

        self.mask_id = 151666 
        self.eos_id = 151643
        
        raw_use_hts = kwargs.get("use_hts", False)
        if isinstance(raw_use_hts, str):
            self.use_hts = raw_use_hts.lower() == "true"
        else:
            self.use_hts = bool(raw_use_hts)

        self.realtime_output = kwargs.get("realtime_output", "eval_results.jsonl")

        if self.use_hts:
            from .hts_sampler import HTSSampler
            self.hts_sampler = HTSSampler(self.model, self.tokenizer, device=self.device)
            eval_logger.info(f"Rank {self.rank}: HTS Sampler initialized for Dream.")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        self.model = (
            transformers.AutoModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str], gen_kwargs: dict = None) -> Tuple[List[str], List[dict]]:
        raw_val = gen_kwargs.get("use_hts", self.use_hts)
        use_hts_now = str(raw_val).lower() == "true" if not isinstance(raw_val, bool) else raw_val
        
        all_stats = []
        if not use_hts_now:
            prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
            prompt_ids = prompt_ids[:, -self.max_prompt_len:]
            attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id).to(self.device)
            prompt_ids = prompt_ids.to(device=self.device)

            generation_ids = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
            )
            responses = [
                self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
                for p, g in zip(prompt_ids, generation_ids.sequences)
            ]
            all_stats = [{} for _ in responses]
            return responses, all_stats
        else:
            if not hasattr(self, "hts_sampler"):
                from .hts_sampler import HTSSampler
                self.hts_sampler = HTSSampler(self.model, self.tokenizer, device=self.device)

            results = []
            for prompt in prompts:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                
                final_codes, stats = self.hts_sampler.generate_hts(
                    prompt_text=prompt,
                    input_ids=input_ids,
                    initial_N=int(gen_kwargs.get("initial_N", 4)),
                    final_K=int(gen_kwargs.get("final_K", 1)),
                    hts_survivor_k=int(gen_kwargs.get("hts_survivor_k", 4)),
                    reward_mode=gen_kwargs.get("reward_mode", "svf"),
                    task_type=gen_kwargs.get("task_type", "code"),
                    steps=self.diffusion_steps,
                    gen_length=self.max_new_tokens,
                    temperature=float(gen_kwargs.get("temperature", self.temperature)),
                    top_p=float(gen_kwargs.get("top_p", self.top_p)),
                    top_k=gen_kwargs.get("top_k", self.top_k),
                    until=gen_kwargs.get("until", []),
                    hts_mode=True,
                    mask_id=self.mask_id,
                    eos_id=self.eos_id
                )
                
                results.append(final_codes[0])
                all_stats.append(stats)
            return results, all_stats

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        
        gen_kwargs_first = requests[0].args[1]
        actual_output_path = gen_kwargs_first.get("realtime_output", self.realtime_output)

        raw_val = gen_kwargs_first.get("use_hts", self.use_hts)
        self.use_hts = str(raw_val).lower() == "true" if not isinstance(raw_val, bool) else raw_val
        
        rank_tmp_file = actual_output_path.replace(".jsonl", f"_rank{self.rank}.tmp")
        
        output_dir = os.path.dirname(rank_tmp_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until",
        )

        for batch_idx in range(0, len(requests), self.batch_size_per_gpu):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size_per_gpu]
            contexts, task_gen_args = zip(*[req.arguments for req in batch_requests])
            
            responses, stats_list = self._generate_batch(contexts, gen_kwargs=task_gen_args[0])

            for i, r in enumerate(responses):
                r = r.replace("```python", "").replace("```", "")
                
                for s in task_gen_args[0].get('until', []):
                    r = r.split(s)[0]
                
                target_val = getattr(batch_requests[i], "target", None)
                if target_val is None or target_val == "N/A":
                    target_val = batch_requests[i].doc.get("answer", batch_requests[i].doc.get("solution", "N/A"))

                save_data = {
                    "doc": batch_requests[i].doc,
                    "target": target_val,
                    "prompt": contexts[i],
                    "response": r,
                }
                
                if self.use_hts:
                    save_data.update(stats_list[i])
                
                with open(rank_tmp_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
                    f.flush()

                responses[i] = r

            if self.rank == 0 and batch_idx == 0:
                print(f"Sample Response:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(batch_requests))

        pbar.close()

        self.accelerator.wait_for_everyone()
        
        if self.rank == 0:
            eval_logger.info(f"Merging rank files into {actual_output_path}...")
            with open(actual_output_path, "w", encoding="utf-8") as final_f:
                for r in range(self.world_size):
                    temp_f = actual_output_path.replace(".jsonl", f"_rank{r}.tmp")
                    if os.path.exists(temp_f):
                        with open(temp_f, "r", encoding="utf-8") as tf:
                            for line in tf:
                                final_f.write(line)
                        os.remove(temp_f)
            eval_logger.info("Merge completed.")

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1
        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)
        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False
        noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.pad_to_max_len:
            raise NotImplementedError
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input, 'full').logits

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.classifier_free_guidance * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        mc_num = self.diffusion_steps
        for _ in range(max(mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
            del logits, loss, perturbed_seq, perturbed_seq_, p_mask, mask_indices
            empty_cache_by_memory(threshold_gb=70)

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = False
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError