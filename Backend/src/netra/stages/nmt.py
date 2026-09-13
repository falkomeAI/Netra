from __future__ import annotations

import re
from typing import Any

import torch

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry

INDIC_LANG_MAP = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "en": "eng_Latn",
    "sa": "san_Deva",
    "ne": "npi_Deva",
    "kok": "gom_Deva",
    "mai": "mai_Deva",
    "sd": "snd_Arab",
    "doi": "doi_Deva",
    "mni": "mni_Beng",
    "brx": "brx_Deva",
    "ks": "kas_Arab",
    "sat": "sat_Olck",
}


@StageRegistry.register("nmt", variant="indictrans2_1b")
@StageRegistry.register("nmt", variant="indictrans2_200m")
@StageRegistry.register("nmt", variant="indictrans2_320m")
class IndicTransNMTStage(BaseStage):
    """
    Neural Machine Translation using AI4Bharat IndicTrans2.
    SOTA for English <-> Indian languages and Indic <-> Indic.
    """

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = self.config["model_id"]
        params = self.config.get("params", {})

        is_cpu = "cpu" in self.device
        dtype = torch.float32 if is_cpu else getattr(torch, params.get("torch_dtype", "float16"), torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=params.get("trust_remote_code", True),
        )
        target_device = "cpu" if is_cpu else self.device
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=params.get("trust_remote_code", True),
        ).to(target_device)

        self._beam_size = params.get("beam_size", 5)

    def _unload_model(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("llm_output", {}).get("response", "")
        if not text:
            text = inputs.get("text", "")
        if not text:
            raise ValueError("No text provided for translation")

        src_lang = inputs.get("source_language", "en")
        tgt_lang = inputs.get("target_language", "hi")

        if src_lang == "auto":
            from netra.utils.language import detect_language
            src_lang = detect_language(text)

        if src_lang == tgt_lang:
            return {
                "translated_text": text,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "skipped": True,
            }

        src_code = INDIC_LANG_MAP.get(src_lang, src_lang)
        tgt_code = INDIC_LANG_MAP.get(tgt_lang, tgt_lang)

        self._tokenizer.src_lang = src_code

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                num_beams=self._beam_size,
                max_new_tokens=512,
            )

        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

        return {
            "translated_text": translated[0] if translated else "",
            "source_language": src_lang,
            "target_language": tgt_lang,
        }


@StageRegistry.register("nmt", variant="nllb_600m")
class NLLBTranslationStage(BaseStage):
    """
    Translation using Meta's NLLB-200 for broader language support.
    Fully supports CPU inference.
    """

    NLLB_LANG_MAP = {
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "bn": "ben_Beng",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "pa": "pan_Guru",
        "or": "ory_Orya",
        "en": "eng_Latn",
        "as": "asm_Beng",
        "ur": "urd_Arab",
        "sa": "san_Deva",
        "ne": "npi_Deva",
        "sd": "snd_Arab",
        "kok": "gom_Deva",
        "mai": "mai_Deva",
        "doi": "doi_Deva",
        "mni": "mni_Beng",
        "brx": "brx_Deva",
        "ks": "kas_Arab",
        "sat": "sat_Olck",
    }

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = self.config["model_id"]
        params = self.config.get("params", {})

        is_cpu = "cpu" in self.device
        dtype = torch.float32 if is_cpu else getattr(torch, params.get("torch_dtype", "float16"), torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        target_device = "cpu" if is_cpu else self.device
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=dtype,
        ).to(target_device)

    def _unload_model(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("llm_output", {}).get("response", "")
        if not text:
            text = inputs.get("text", "")
        if not text:
            raise ValueError("No text provided for translation")

        src_lang = inputs.get("source_language", "en")
        tgt_lang = inputs.get("target_language", "hi")

        if src_lang == "auto":
            from netra.utils.language import detect_language
            src_lang = detect_language(text)

        if src_lang == tgt_lang:
            return {
                "translated_text": text,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "skipped": True,
            }

        src_code = self.NLLB_LANG_MAP.get(src_lang, src_lang)
        tgt_code = self.NLLB_LANG_MAP.get(tgt_lang, tgt_lang)

        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=512,
            )

        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

        return {
            "translated_text": translated[0] if translated else "",
            "source_language": src_lang,
            "target_language": tgt_lang,
        }


_NMT_LANG_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "mr": "Marathi",
    "ta": "Tamil", "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada",
    "or": "Odia", "ml": "Malayalam", "pa": "Punjabi", "as": "Assamese",
    "mai": "Maithili", "sa": "Sanskrit", "kok": "Konkani", "ne": "Nepali",
    "sd": "Sindhi", "doi": "Dogri", "mni": "Manipuri", "brx": "Bodo",
    "sat": "Santali", "ks": "Kashmiri", "en": "English",
}


@StageRegistry.register("nmt", variant="ollama_nmt")
class OllamaNMTStage(BaseStage):
    """
    Translation using Ollama LLM — prompts the model to translate text
    between languages. Works with any Ollama chat model (qwen2.5, llama3, etc.).
    """

    def _load_model(self) -> None:
        import urllib.request
        import json

        params = self.config.get("params", {})
        self._ollama_url = params.get("ollama_url", "http://localhost:11434")
        self._ollama_model = self.config.get("ollama_model", "qwen2.5:3b")

        try:
            req = urllib.request.Request(f"{self._ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            self.logger.warning("Ollama not reachable at %s: %s", self._ollama_url, e)

        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import urllib.request
        import json

        text = inputs.get("llm_output", {}).get("response", "")
        if not text:
            text = inputs.get("text", "")
        if not text:
            raise ValueError("No text provided for translation")

        src_lang = inputs.get("source_language", "en")
        tgt_lang = inputs.get("target_language", "hi")

        if src_lang == "auto" or src_lang == tgt_lang:
            try:
                from langdetect import detect as _ld
                detected = _ld(text[:500])
                if detected and detected != tgt_lang:
                    src_lang = detected
                    self.logger.info("NMT: auto-detected source language as %s", src_lang)
            except Exception:
                pass

        if src_lang == tgt_lang:
            return {
                "translated_text": text,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "skipped": True,
            }

        src_name = _NMT_LANG_NAMES.get(src_lang, src_lang)
        tgt_name = _NMT_LANG_NAMES.get(tgt_lang, tgt_lang)

        is_thinking_model = "qwen3" in self._ollama_model.lower()
        no_think_prefix = "/no_think\n" if is_thinking_model else ""

        system_prompt = (
            f"{no_think_prefix}You are a professional translator. Translate the following text from "
            f"{src_name} to {tgt_name}. Output ONLY the translated text, nothing else. "
            f"Do not add explanations, notes, or formatting."
        )

        payload = json.dumps({
            "model": self._ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.3, "num_predict": 512},
        }).encode()

        req = urllib.request.Request(
            f"{self._ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"Ollama translation failed: {e}") from e

        message = result.get("message", {})
        translated = message.get("content", "").strip()
        thinking = message.get("thinking", "")

        if not translated and thinking:
            lines = thinking.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line and not re.match(
                    r"^(Okay|Wait|Hmm|Let me|First|I need|I'll|I should|The user|Looking|Now|So |But |Actually|However)",
                    line, re.IGNORECASE,
                ):
                    translated = line
                    break

        if translated:
            translated = re.sub(
                r"^(Okay|Wait|Hmm|Let me|First|I need|I'll|So |Now |But |Actually|However|Looking).*?\n",
                "", translated, flags=re.IGNORECASE | re.MULTILINE,
            ).strip()

        return {
            "translated_text": translated,
            "source_language": src_lang,
            "target_language": tgt_lang,
        }
