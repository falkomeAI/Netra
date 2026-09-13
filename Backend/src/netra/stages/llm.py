from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

import torch

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry

LANG_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "mr": "Marathi",
    "ta": "Tamil", "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada",
    "or": "Odia", "ml": "Malayalam", "pa": "Punjabi", "as": "Assamese",
    "mai": "Maithili", "sa": "Sanskrit", "kok": "Konkani", "ne": "Nepali",
    "sd": "Sindhi", "doi": "Dogri", "mni": "Manipuri", "brx": "Bodo",
    "sat": "Santali", "ks": "Kashmiri", "en": "English",
}


_MAX_CONTEXT_CHARS = 500

def _truncate_context(text: str, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Truncate long context to keep prompt eval fast on edge devices."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for speed]"


def _resolve_prompt_placeholders(system_prompt: str, user_prompt: str, inputs: dict[str, Any]) -> tuple[str, str]:
    """Shared logic to resolve template placeholders in prompts."""
    ocr_text = inputs.get("ocr_output", {}).get("text", "")
    rag_context = inputs.get("retrieval_output", {}).get("context_text", "")

    ocr_text = _truncate_context(ocr_text)
    rag_context = _truncate_context(rag_context)

    if "{document_text}" in user_prompt or "{document_text}" in system_prompt:
        doc_text = ocr_text or rag_context or "No document available."
        user_prompt = user_prompt.replace("{document_text}", doc_text)
        system_prompt = system_prompt.replace("{document_text}", doc_text)

    if "{rag_context}" in user_prompt or "{rag_context}" in system_prompt:
        ctx = rag_context or "No knowledge base context available."
        user_prompt = user_prompt.replace("{rag_context}", ctx)
        system_prompt = system_prompt.replace("{rag_context}", ctx)
    elif rag_context and ocr_text:
        user_prompt = f"Related context from knowledge base:\n{rag_context}\n\n{user_prompt}"

    target_lang_code = inputs.get("target_language", "hi")
    target_lang_name = LANG_NAMES.get(target_lang_code, target_lang_code)
    if "{target_language}" in user_prompt:
        user_prompt = user_prompt.replace("{target_language}", target_lang_name)
    if "{target_language}" in system_prompt:
        system_prompt = system_prompt.replace("{target_language}", target_lang_name)

    question = inputs.get("question", "")
    if "{question}" in user_prompt:
        user_prompt = user_prompt.replace("{question}", question)

    return system_prompt, user_prompt


import re

_REASONING_PATTERNS = re.compile(
    r"^(Okay|Wait|Hmm|Let me|First|I need|I'll|I should|The user|"
    r"Looking at|Now|So |But |Actually|However|Alright)",
    re.IGNORECASE,
)


def _strip_reasoning_prefix(text: str) -> str:
    """Remove internal reasoning/thinking that leaked into the visible response."""
    if not text:
        return text
    lines = text.split("\n")
    clean = []
    reasoning_done = False
    for line in lines:
        stripped = line.strip()
        if not reasoning_done:
            if not stripped:
                continue
            if _REASONING_PATTERNS.match(stripped):
                continue
            reasoning_done = True
        clean.append(line)
    result = "\n".join(clean).strip()
    return result if result else text


def _extract_answer_from_thinking(thinking: str) -> str:
    """Extract the most answer-like content from thinking text."""
    lines = thinking.strip().split("\n")
    answer_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not _REASONING_PATTERNS.match(stripped):
            answer_lines.insert(0, line)
            if len(answer_lines) >= 10:
                break
    return "\n".join(answer_lines).strip() if answer_lines else thinking[-1000:].strip()


@StageRegistry.register("llm", variant="ollama_qwen2_5_1_5b")
@StageRegistry.register("llm", variant="ollama_qwen3_4b")
@StageRegistry.register("llm", variant="ollama_qwen2_5_3b")
@StageRegistry.register("llm", variant="ollama_qwen2_5_7b")
@StageRegistry.register("llm", variant="ollama_llama3_2_3b")
@StageRegistry.register("llm", variant="ollama_gemma2_2b")
@StageRegistry.register("llm", variant="ollama_phi3_5_mini")
@StageRegistry.register("llm", variant="ollama_mistral")
@StageRegistry.register("llm", variant="ollama_custom")
class OllamaLLMStage(BaseStage):
    """
    LLM inference via local Ollama server (HTTP API).
    Ollama handles quantization, memory management, and GPU offloading natively.
    """

    def _load_model(self) -> None:
        params = self.config.get("params", {})
        self._ollama_url = params.get("ollama_url", "http://localhost:11434")
        self._model_name = self.config.get("ollama_model", self.config.get("model_id", "qwen2.5:3b"))
        self._max_new_tokens = params.get("max_new_tokens", 512)
        self._temperature = params.get("temperature", 0.7)
        self._top_p = params.get("top_p", 0.9)

        try:
            req = urllib.request.Request(
                f"{self._ollama_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"] for m in data.get("models", [])]
                self.logger.info("Ollama server online, available models: %s", available)
                if self._model_name not in available and f"{self._model_name}:latest" not in available:
                    self.logger.warning("Model '%s' not found locally, Ollama will pull on first use", self._model_name)
        except Exception as e:
            self.logger.warning("Could not connect to Ollama at %s: %s", self._ollama_url, e)

        self._model = True  # sentinel

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        system_prompt, user_prompt = self._build_prompt(inputs)

        num_predict = min(
            inputs.get("max_new_tokens", self._max_new_tokens),
            self._max_new_tokens,
        )

        is_thinking_model = "qwen3" in self._model_name.lower()

        if is_thinking_model and system_prompt:
            system_prompt = "/no_think\n" + system_prompt

        payload = {
            "model": self._model_name,
            "messages": [],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": num_predict,
                "temperature": inputs.get("temperature", self._temperature),
                "top_p": inputs.get("top_p", self._top_p),
            },
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": user_prompt})

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._ollama_url}/api/chat",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama server unreachable at {self._ollama_url}: {e}") from e

        message = result.get("message", {})
        response = message.get("content", "").strip()
        thinking = message.get("thinking", "")

        if not response and thinking:
            self.logger.info("Thinking mode fallback — extracting from thinking field")
            response = _extract_answer_from_thinking(thinking)

        response = _strip_reasoning_prefix(response)

        prompt_tokens = result.get("prompt_eval_count", 0)
        completion_tokens = result.get("eval_count", 0)

        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _build_prompt(self, inputs: dict[str, Any]) -> tuple[str, str]:
        system_prompt = inputs.get("system_prompt", "")
        user_prompt = inputs.get("user_prompt", "")
        system_prompt, user_prompt = _resolve_prompt_placeholders(system_prompt, user_prompt, inputs)
        return system_prompt, user_prompt


@StageRegistry.register("llm", variant="ollm_llama3_3b")
@StageRegistry.register("llm", variant="ollm_llama3_8b")
@StageRegistry.register("llm", variant="ollm_gemma3_12b")
@StageRegistry.register("llm", variant="ollm_qwen3_next_80b")
@StageRegistry.register("llm", variant="ollm_gpt_oss_20b")
class OLLMStage(BaseStage):
    """
    LLM inference via oLLM — runs large models on consumer GPUs (8GB VRAM)
    by offloading layer weights and KV cache to SSD.
    Supports: llama3-1B/3B/8B-chat, gemma3-12B, qwen3-next-80B, gpt-oss-20B.
    See: https://github.com/Mega4alik/ollm
    """

    OLLM_MODEL_MAP = {
        "ollm_llama3_1b": "llama3-1B-chat",
        "ollm_llama3_3b": "llama3-3B-chat",
        "ollm_llama3_8b": "llama3-8B-chat",
        "ollm_gemma3_12b": "gemma3-12B",
        "ollm_qwen3_next_80b": "qwen3-next-80B",
        "ollm_gpt_oss_20b": "gpt-oss-20B",
    }

    def _load_model(self) -> None:
        from ollm import Inference

        params = self.config.get("params", {})
        variant = self.config.get("_variant", "")
        ollm_model_id = self.config.get("ollm_model_id") or self.OLLM_MODEL_MAP.get(variant, "llama3-3B-chat")
        models_dir = params.get("models_dir", "models/ollm")
        cache_dir = params.get("kv_cache_dir", "cache/ollm_kv")
        cpu_offload_layers = params.get("cpu_offload_layers", 2)

        self._ollm = Inference(ollm_model_id, device=self.device, logging=False)
        self._ollm.ini_model(models_dir=models_dir, force_download=False)

        if cpu_offload_layers > 0:
            self._ollm.offload_layers_to_cpu(layers_num=cpu_offload_layers)

        use_disk_cache = params.get("use_disk_cache", True)
        self._past_kv = self._ollm.DiskCache(cache_dir=cache_dir) if use_disk_cache else None

        self._max_new_tokens = params.get("max_new_tokens", 512)
        self._model = self._ollm.model

    def _unload_model(self) -> None:
        del self._ollm
        del self._model
        self._ollm = None
        self._model = None
        self._past_kv = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        system_prompt, user_prompt = self._build_prompt(inputs)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        max_new_tokens = inputs.get("max_new_tokens", self._max_new_tokens)

        input_ids = self._ollm.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._ollm.device)

        prompt_tokens = input_ids.shape[1]

        with torch.inference_mode():
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": inputs.get("temperature", 0.7) > 0,
                "temperature": inputs.get("temperature", 0.7),
                "top_p": inputs.get("top_p", 0.9),
                "pad_token_id": self._ollm.tokenizer.eos_token_id,
            }
            if self._past_kv is not None:
                gen_kwargs["past_key_values"] = self._past_kv

            output_ids = self._ollm.model.generate(**gen_kwargs).cpu()

        new_tokens = output_ids[0][prompt_tokens:]
        response = self._ollm.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "response": response.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(new_tokens),
        }

    def _build_prompt(self, inputs: dict[str, Any]) -> tuple[str, str]:
        system_prompt = inputs.get("system_prompt", "")
        user_prompt = inputs.get("user_prompt", "")
        system_prompt, user_prompt = _resolve_prompt_placeholders(system_prompt, user_prompt, inputs)
        return system_prompt, user_prompt


@StageRegistry.register("llm", variant="qwen2_5_3b")
@StageRegistry.register("llm", variant="qwen2_5_05b")
@StageRegistry.register("llm", variant="gemma2_2b")
@StageRegistry.register("llm", variant="phi3_5_mini")
@StageRegistry.register("llm", variant="llama3_2_3b")
@StageRegistry.register("llm", variant="smollm2")
class HuggingFaceLLMStage(BaseStage):
    """
    LLM inference via HuggingFace Transformers.
    Supports AWQ / GPTQ INT4 quantized models and CPU float32 mode.
    """

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config["model_id"]
        params = self.config.get("params", {})
        quant_method = self.config.get("quantization", "none")

        dtype_str = params.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16)

        is_cpu = "cpu" in self.device or params.get("device_map") == "cpu"
        if is_cpu:
            dtype = torch.float32

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": params.get("trust_remote_code", False),
        }

        if is_cpu:
            load_kwargs["device_map"] = "cpu"
        else:
            load_kwargs["device_map"] = params.get("device_map", "auto")

        if quant_method == "gptq" and not is_cpu:
            from transformers import GPTQConfig
            load_kwargs["quantization_config"] = GPTQConfig(
                bits=self.config.get("bits", 4),
                disable_exllama=True,
            )
        elif quant_method == "awq" and not is_cpu:
            from transformers import AwqConfig
            load_kwargs["quantization_config"] = AwqConfig(
                bits=self.config.get("bits", 4),
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=params.get("trust_remote_code", False),
        )
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self._max_new_tokens = params.get("max_new_tokens", 512)

    def _unload_model(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(inputs)

        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        temperature = inputs.get("temperature", 0.7)
        top_p = inputs.get("top_p", 0.9)
        max_new_tokens = inputs.get("max_new_tokens", self._max_new_tokens)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][encoded["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "response": response.strip(),
            "prompt_tokens": encoded["input_ids"].shape[1],
            "completion_tokens": len(new_tokens),
        }

    def _build_prompt(self, inputs: dict[str, Any]) -> str:
        system_prompt = inputs.get("system_prompt", "")
        user_prompt = inputs.get("user_prompt", "")
        system_prompt, user_prompt = _resolve_prompt_placeholders(system_prompt, user_prompt, inputs)

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(f"User: {user_prompt}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
