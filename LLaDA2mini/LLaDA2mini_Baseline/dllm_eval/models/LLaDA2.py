import logging
import os
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union, TypeVar
import torch
import torch.nn.functional as F
import numpy as np
import transformers
import json
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from packaging import version
from tqdm import tqdm
import torch.distributed as dist
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from dllm_eval.api.instance import Instance
from dllm_eval.api.model import LM, TemplateLM
from dllm_eval.api.registry import register_model
from dllm_eval.models.utils import get_dtype, configure_pad_token

try:
    from .hts_sampler import HTSSampler
except ImportError:
    HTSSampler = None

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


@register_model("LLaDA2")
class LLaDA2(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModel
    _DEFAULT_MAX_LENGTH = 20480
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until:Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        mc_num: int = 1024,
        remasking: str = "low_confidence",
        mask_id: int = 156895,
        is_check_greedy : bool =True,
        assistant_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.is_check_greedy = is_check_greedy
        self.assistant_prefix = assistant_prefix
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))
            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator
            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()
            if not (parallelize or accelerator.num_processes > 1):
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
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )
            revision = str(revision)
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )
        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > self.accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {self.accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{self.accelerator.device}")
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
                
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        self.is_first_inference = True

        if HTSSampler is not None:
            self.hts_sampler = HTSSampler(self.model, self.tokenizer, device=self.device)
            eval_logger.info("HTSSampler initialized successfully.")
        
    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        if hasattr(self, "accelerator"):
            return self.accelerator.local_process_index
        return int(os.environ.get("LOCAL_RANK", 0))
    
    @property
    def world_size(self):
        if hasattr(self, "_world_size"):
            return self._world_size
        if hasattr(self, "accelerator"):
            return self.accelerator.num_processes
        return int(os.environ.get("WORLD_SIZE", 1))
    
    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if parallelize is None and gpus is not None and gpus > 1:
            parallelize = True
        args = {}
        if parallelize:
            max_memory_all_gpus = get_max_memory()
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_memory_per_gpu_map = {
                device_idx: max_memory_per_gpu for device_idx in range(len(max_memory_all_gpus))
            } if max_memory_per_gpu is not None else {k: v for k, v in max_memory_all_gpus.items()}
            if hasattr(self, "accelerator"):
                max_memory_per_gpu_map = {
                    k: v for k, v in max_memory_all_gpus.items() if k % num_local_processes == self.accelerator.process_index % num_local_processes
                }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto"
            args["offload_folder"] = offload_folder
            if max_cpu_memory is not None:
                args["max_memory"]["cpu"] = max_cpu_memory
            eval_logger.info(
                f"Model parallel set to True. Max memory per GPU: {args['max_memory']}, Device map: {args['device_map']}"
            )
        else:
            args["device_map"] = {"": str(self.device)}
            eval_logger.info(
                f"Model parallel set to False. Device map: {args['device_map']}"
            )
        return args

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length > 1e10:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]
        if backend != "default":
            self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "seq2seq"
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
            else:
                eval_logger.warning(
                    "HF model type is neither CausalLM nor Seq2SeqLM. Assuming CausalLM."
                )
                self.backend = "causal"

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        if autogptq or gptqmodel:
            raise NotImplementedError("Quantization options are not implemented for this custom class.")
        model_dtype = get_dtype(dtype)
        eval_logger.info(f"Loading model with dtype: {model_dtype}")
        model_kwargs = kwargs if kwargs else {}
        if not parallelize:
            model_kwargs.update(
                self._get_accelerate_args(
                    parallelize=parallelize,
                    gpus=gpus,
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                )
            )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        if peft:
            from peft import PeftModel
            eval_logger.info(f"Loading PEFT model from {peft}")
            self._model = PeftModel.from_pretrained(self._model, peft, torch_dtype=model_dtype)
        if not parallelize:
            self._model = self._model.to(self.device)
        self._model = self._model.to(torch.bfloat16)
        self._model.eval()

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_tokenizer
        }
        if add_bos_token:
            kwargs["add_bos_token"] = True
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
            else:
                self.tokenizer = tokenizer
        else:
            model_name = pretrained if isinstance(pretrained, str) else self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs["add_special_tokens"] = self.add_bos_token
        else:
            special_tokens_kwargs["add_special_tokens"] = add_special_tokens
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = {"add_special_tokens": self.add_bos_token} if self.backend == "causal" else {}
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and encoding["input_ids"].size(1) > left_truncate_len:
            eval_logger.warning(
                f"Left-truncating from {encoding['input_ids'].size(1)} to {left_truncate_len} tokens."
            )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if self.backend == "seq2seq":
                return self.model(input_ids=inps, attention_mask=attn_mask, labels=labels).logits
            else:
                return self.model(inps, attention_mask=attn_mask).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError
         
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        gen_kwargs = requests[0].args[1]
        use_hts = gen_kwargs.get("use_hts", False)
        
        realtime_output = gen_kwargs.get("realtime_output", "realtime_hts_results.jsonl")
        baseline_realtime_output = "realtime_baseline_results.jsonl"

        if not use_hts:
            bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running Baseline")
            ds_data = [{"text": req.args[0]} for req in requests]
            ds = Dataset.from_list(ds_data)
            
            req_idx = 0 
            for batch in ds.iter(batch_size=int(self.batch_size)):
                contexts = batch["text"]
                context_enc, _ = self.tok_batch_encode(contexts)
                prompt_length = context_enc.shape[1]
                
                out_full = self.model.generate(
                    inputs=context_enc,
                    steps=gen_kwargs.get("steps", 32),
                    gen_length=gen_kwargs.get("gen_length", 512),
                    block_length=gen_kwargs.get("block_length", 32),
                    temperature=gen_kwargs.get("temperature", 0.7),
                    eos_early_stop=gen_kwargs.get("eos_early_stop", False),
                )
                generated_tokens = out_full[:, prompt_length:]
                cont_toks_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                for i, s in enumerate(cont_toks_list):
                    s = s.strip()
                    
                    if not self.escape_until:
                        until_terms = gen_kwargs.get("until", [])
                        for term in until_terms:
                            if len(term) > 0 and term in s:
                                s = s.split(term)[0]
                    
                    orig_req = requests[req_idx]
                    target_val = getattr(orig_req, "target", None)
                    if target_val is None or target_val == "N/A":
                        if "test" in orig_req.doc and "entry_point" in orig_req.doc:
                            target_val = orig_req.doc["test"] + "\ncheck(" + orig_req.doc["entry_point"] + ")"
                        else:
                            target_val = orig_req.doc.get("answer", orig_req.doc.get("solution", "N/A"))
                        
                    with open(baseline_realtime_output, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "doc": orig_req.doc, 
                            "target": target_val, 
                            "resps": [[s]], 
                            "prompt": contexts[i]
                        }, ensure_ascii=False) + "\n")
                        f.flush()
                    
                    res.append(s)
                    bar.update(1)
                    req_idx += 1
            bar.close()
            
        else:
            bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running HTS+SVF")
            for req in requests:
                prompt_text = req.args[0]
                context_enc, _ = self.tok_batch_encode([prompt_text])
                
                p_interval = int(gen_kwargs.get("pruning_interval", 0))

                final_codes, stats = self.hts_sampler.generate_hts(
                    prompt_text=prompt_text,
                    input_ids=context_enc,
                    initial_N=int(gen_kwargs.get("hts_N", 4)),
                    final_K=int(gen_kwargs.get("final_K", 1)),
                    hts_survivor_k=int(gen_kwargs.get("hts_survivor_k", 4)),
                    hts_mode=gen_kwargs.get("hts_mode", True),
                    hts_start_pct=float(gen_kwargs.get("hts_start_pct", 0.1)),
                    hts_end_pct=float(gen_kwargs.get("hts_end_pct", 0.6)),
                    decay_factor=float(gen_kwargs.get("decay_factor", 1.5)),
                    pruning_interval=p_interval,
                    reward_mode=gen_kwargs.get("reward_mode", "svf"),
                    task_type=gen_kwargs.get("task_type", "code"),
                    steps=int(gen_kwargs.get("steps", 32)),
                    gen_length=int(gen_kwargs.get("gen_length", 512)),
                    block_length=int(gen_kwargs.get("block_length", 32)),
                    temperature=float(gen_kwargs.get("temperature", 0.7)),
                    top_p=float(gen_kwargs.get("top_p", 0.95)),
                    top_k=gen_kwargs.get("top_k", None),
                    threshold=float(gen_kwargs.get("threshold", 0.85)),
                    mask_id=self.mask_id,
                    eos_id=self.eot_token_id
                )
                
                processed_codes = []
                for code in final_codes:
                    code = code.strip()
                    if not self.escape_until:
                        until_terms = gen_kwargs.get("until", [])
                        for term in until_terms:
                            if len(term) > 0 and term in code:
                                code = code.split(term)[0]
                    processed_codes.append(code)
                
                final_choice = processed_codes[0]
                res.append(final_choice)

                target_val = getattr(req, "target", None)
                if target_val is None or target_val == "N/A":
                    if "test" in req.doc and "entry_point" in req.doc:
                        target_val = req.doc["test"] + "\ncheck(" + req.doc["entry_point"] + ")"
                    else:
                        target_val = req.doc.get("answer", req.doc.get("solution", "N/A"))

                with open(realtime_output, "a", encoding="utf-8") as f:
                    all_resps = [[code] for code in processed_codes]
                    
                    output_data = {
                        "doc": req.doc, 
                        "target": target_val, 
                        "resps": all_resps, 
                        "prompt": prompt_text,
                        "entropy_history": stats.get("entropy_history", []),
                        "pruning_history": stats.get("pruning_history", []),
                        "final_scores": stats.get("final_scores", []),
                        "all_trajectories": stats.get("all_trajectories", []), 
                        "nfe": stats.get("nfe", 0),
                        "svf_calls": stats.get("svf_calls", 0),
                        "total_steps": stats.get("total_steps", 0)
                    }
                    f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    f.flush()
                
                bar.update(1)
            bar.close()
            
        return res
     
    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.assistant_prefix:
            chat_templated += self.assistant_prefix
        return chat_templated