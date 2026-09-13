from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry

logger = logging.getLogger("netra.tts")


@StageRegistry.register("tts", variant="indic_tts")
class IndicTTSStage(BaseStage):
    """
    Text-to-Speech using AI4Bharat IndicTTS (VITS-based).
    Covers all 22 Indian scheduled languages.
    """

    def _load_model(self) -> None:
        from TTS.api import TTS

        model_id = self.config.get("model_id", "ai4bharat/indic-tts-coqui")
        params = self.config.get("params", {})

        gpu = "cuda" in self.device
        self._model = TTS(model_name=model_id, gpu=gpu)
        self._sample_rate = params.get("sample_rate", 22050)
        self._speaker_id = params.get("speaker_id")

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = self._extract_text(inputs)
        if not text:
            raise ValueError("No text provided for TTS")

        output_path = inputs.get("tts_output_path")
        if not output_path:
            output_dir = inputs.get("output_dir", "output")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / "speech.wav")

        kwargs = {}
        if self._speaker_id is not None:
            kwargs["speaker"] = self._speaker_id

        self._model.tts_to_file(text=text, file_path=output_path, **kwargs)

        return {
            "audio_path": output_path,
            "sample_rate": self._sample_rate,
            "text_length": len(text),
        }

    @staticmethod
    def _extract_text(inputs: dict[str, Any]) -> str:
        nmt_output = inputs.get("nmt_output", {})
        if nmt_output.get("translated_text"):
            return nmt_output["translated_text"]

        llm_output = inputs.get("llm_output", {})
        if llm_output.get("response"):
            return llm_output["response"]

        return inputs.get("text", "")


@StageRegistry.register("tts", variant="piper")
class PiperTTSStage(BaseStage):
    """
    Lightweight TTS using Piper — ideal for resource-constrained Jetson deployments.
    """

    def _load_model(self) -> None:
        from piper import PiperVoice

        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("Piper TTS requires 'model_path' in config")

        self._model = PiperVoice.load(model_path)
        params = self.config.get("params", {})
        self._sample_rate = params.get("sample_rate", 22050)
        self._length_scale = params.get("length_scale", 1.0)
        self._noise_scale = params.get("noise_scale", 0.667)
        self._noise_w = params.get("noise_w", 0.8)

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = self._extract_text(inputs)
        if not text:
            raise ValueError("No text provided for TTS")

        output_path = inputs.get("tts_output_path")
        if not output_path:
            output_dir = inputs.get("output_dir", "output")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / "speech.wav")

        with wave.open(output_path, "wb") as wf:
            self._model.synthesize_wav(text, wf)
        self._sample_rate = self._model.config.sample_rate

        return {
            "audio_path": output_path,
            "sample_rate": self._sample_rate,
            "text_length": len(text),
        }

    @staticmethod
    def _extract_text(inputs: dict[str, Any]) -> str:
        nmt_output = inputs.get("nmt_output", {})
        if nmt_output.get("translated_text"):
            return nmt_output["translated_text"]
        llm_output = inputs.get("llm_output", {})
        if llm_output.get("response"):
            return llm_output["response"]
        return inputs.get("text", "")

    @staticmethod
    def _save_wav(path: str, raw_audio: bytes, sample_rate: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)


@StageRegistry.register("tts", variant="edge_tts")
class EdgeTTSStage(BaseStage):
    """
    Lightweight TTS using Microsoft Edge TTS.
    No GPU required — works via edge-tts library.
    Supports all major Indian languages.
    """

    DEFAULT_VOICE_MAP = {
        "hi": "hi-IN-SwaraNeural",
        "ta": "ta-IN-PallaviNeural",
        "te": "te-IN-ShrutiNeural",
        "bn": "bn-IN-TanishaaNeural",
        "mr": "mr-IN-AarohiNeural",
        "gu": "gu-IN-DhwaniNeural",
        "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-SobhanaNeural",
        "pa": "pa-IN-Neural2-A",
        "ur": "ur-IN-GulNeural",
        "as": "bn-IN-TanishaaNeural",
        "or": "hi-IN-SwaraNeural",
        "ne": "hi-IN-SwaraNeural",
        "sa": "hi-IN-SwaraNeural",
        "mai": "hi-IN-SwaraNeural",
        "kok": "hi-IN-SwaraNeural",
        "doi": "hi-IN-SwaraNeural",
        "sd": "ur-IN-GulNeural",
        "ks": "ur-IN-GulNeural",
        "mni": "bn-IN-TanishaaNeural",
        "brx": "hi-IN-SwaraNeural",
        "sat": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural",
    }

    def _load_model(self) -> None:
        import edge_tts  # noqa: F401
        params = self.config.get("params", {})
        self._voice_map = params.get("voice_map", self.DEFAULT_VOICE_MAP)
        self._sample_rate = params.get("sample_rate", 24000)
        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import edge_tts

        text = self._extract_text(inputs)
        if not text:
            raise ValueError("No text provided for TTS")

        target_lang = inputs.get("target_language", "hi")
        voice = self._voice_map.get(target_lang, self._voice_map.get("en", "en-IN-NeerjaNeural"))

        output_path = inputs.get("tts_output_path")
        if not output_path:
            output_dir = inputs.get("output_dir", "output")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / "speech.mp3")

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_synthesize())).result()
        else:
            asyncio.run(_synthesize())

        return {
            "audio_path": output_path,
            "voice": voice,
            "sample_rate": self._sample_rate,
            "text_length": len(text),
        }

    @staticmethod
    def _extract_text(inputs: dict[str, Any]) -> str:
        nmt_output = inputs.get("nmt_output", {})
        if nmt_output.get("translated_text"):
            return nmt_output["translated_text"]
        llm_output = inputs.get("llm_output", {})
        if llm_output.get("response"):
            return llm_output["response"]
        return inputs.get("text", "")


INDICF5_LANG_MAP = {
    "as": "assamese", "bn": "bengali", "gu": "gujarati", "hi": "hindi",
    "kn": "kannada", "ml": "malayalam", "mr": "marathi", "or": "odia",
    "pa": "punjabi", "ta": "tamil", "te": "telugu",
}


@StageRegistry.register("tts", variant="indicf5")
class IndicF5TTSStage(BaseStage):
    """
    Near-human TTS using AI4Bharat IndicF5 (F5-TTS architecture).
    Supports 11 Indian languages with natural-sounding speech.
    Model: https://huggingface.co/ai4bharat/IndicF5
    """

    def _load_model(self) -> None:
        from transformers import AutoModel

        model_id = self.config.get("model_id", "ai4bharat/IndicF5")
        params = self.config.get("params", {})
        self._sample_rate = params.get("sample_rate", 24000)

        logger.info(f"Loading IndicF5 from: {model_id}")
        self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        device = "cuda" if "cuda" in self.device else "cpu"
        self._model = self._model.to(device)
        self._model.eval()

    def _unload_model(self) -> None:
        del self._model
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        import soundfile as sf

        text = self._extract_text(inputs)
        if not text:
            raise ValueError("No text provided for TTS")

        target_lang = inputs.get("target_language", "hi")
        lang_name = INDICF5_LANG_MAP.get(target_lang)

        if not lang_name:
            logger.warning(f"IndicF5 does not support '{target_lang}', falling back to Hindi")
            lang_name = "hindi"

        output_path = inputs.get("tts_output_path")
        if not output_path:
            output_dir = inputs.get("output_dir", "output")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / "speech.wav")

        with torch.inference_mode():
            audio = self._model.synthesize(text=text, language=lang_name)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        sf.write(output_path, audio, self._sample_rate)

        return {
            "audio_path": output_path,
            "sample_rate": self._sample_rate,
            "text_length": len(text),
            "language": target_lang,
            "model": "indicf5",
        }

    @staticmethod
    def _extract_text(inputs: dict[str, Any]) -> str:
        nmt_output = inputs.get("nmt_output", {})
        if nmt_output.get("translated_text"):
            return nmt_output["translated_text"]
        llm_output = inputs.get("llm_output", {})
        if llm_output.get("response"):
            return llm_output["response"]
        return inputs.get("text", "")


@StageRegistry.register("tts", variant="espeak")
class EspeakTTSStage(BaseStage):
    """
    Offline TTS using espeak-ng — supports 22 Indian languages.
    No GPU needed, fully offline, zero downloads required.
    """

    LANG_MAP = {
        "hi": "hi", "bn": "bn", "ta": "ta", "te": "te", "mr": "mr",
        "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa", "or": "or",
        "as": "as", "ur": "ur", "sd": "sd", "ks": "ks", "ne": "ne",
        "sa": "sa", "mai": "mai", "doi": "hi", "kok": "hi", "brx": "hi",
        "mni": "bn", "sat": "hi", "en": "en-in",
    }

    def _load_model(self) -> None:
        import subprocess
        result = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("espeak-ng not found on system")
        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import subprocess

        text = self._extract_text(inputs)
        if not text:
            raise ValueError("No text provided for TTS")

        target_lang = inputs.get("target_language", "hi")
        voice = self.LANG_MAP.get(target_lang, "hi")

        output_dir = inputs.get("output_dir", "output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / "speech.wav")

        subprocess.run(
            ["espeak-ng", "-v", voice, "-w", output_path, "--stdin"],
            input=text[:5000], check=True, capture_output=True, text=True,
        )

        return {
            "audio_path": output_path,
            "duration_ms": 0,
            "sample_rate": 22050,
            "voice": voice,
            "engine": "espeak-ng",
        }

    @staticmethod
    def _extract_text(inputs: dict[str, Any]) -> str:
        nmt_output = inputs.get("nmt_output", {})
        if nmt_output.get("translated_text"):
            return nmt_output["translated_text"]
        llm_output = inputs.get("llm_output", {})
        if llm_output.get("response"):
            return llm_output["response"]
        return inputs.get("text", "")
