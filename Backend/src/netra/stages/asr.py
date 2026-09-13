from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry

logger = logging.getLogger("netra.asr")


@StageRegistry.register("asr", variant="indicwhisper")
@StageRegistry.register("asr", variant="whisper_small")
@StageRegistry.register("asr", variant="distil_whisper")
class WhisperASRStage(BaseStage):
    """
    ASR using Whisper-family models via faster-whisper (CTranslate2 backend).
    Supports IndicWhisper, Whisper Small, and Distil-Whisper.
    """

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        model_id = self.config.get("model_id", "Systran/faster-whisper-small")
        params = self.config.get("params", {})
        device = "cuda" if "cuda" in self.device else "cpu"
        compute_type = params.get("compute_type", "int8" if device == "cpu" else "int8_float16")

        self._model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )
        self._params = params

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        audio_path = inputs.get("audio_path")
        if audio_path is None:
            raise ValueError("No audio_path provided for ASR")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        beam_size = self._params.get("beam_size", 5)
        vad_filter = self._params.get("vad_filter", True)
        language = inputs.get("language")

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
        )

        segments_list = []
        full_text_parts = []

        for segment in segments:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segments_list,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }


CONFORMER_LANG_MAP = {
    "as": "as", "bn": "bn", "brx": "brx", "doi": "doi", "en": "en",
    "gu": "gu", "hi": "hi", "kn": "kn", "kok": "kok", "ks": "ks",
    "mai": "mai", "ml": "ml", "mni": "mni", "mr": "mr", "ne": "ne",
    "or": "or", "pa": "pa", "sa": "sa", "sat": "sat", "sd": "sd",
    "ta": "ta", "te": "te", "ur": "ur",
}


@StageRegistry.register("asr", variant="indicconformer")
class IndicConformerASRStage(BaseStage):
    """
    ASR using AI4Bharat IndicConformer (NeMo-based, hybrid RNNT).
    Covers all 22 scheduled Indian languages — best accuracy for Indic ASR.
    Models: https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f
    """

    def _load_model(self) -> None:
        import nemo.collections.asr as nemo_asr

        model_path = self.config.get("model_path")
        model_id = self.config.get("model_id")
        params = self.config.get("params", {})
        self._default_lang = params.get("default_language", "hi")

        if model_path and Path(model_path).exists():
            logger.info(f"Loading IndicConformer from local: {model_path}")
            self._model = nemo_asr.models.ASRModel.restore_from(model_path)
        elif model_id:
            logger.info(f"Loading IndicConformer from HuggingFace: {model_id}")
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_id)
        else:
            raise ValueError("IndicConformer requires 'model_path' or 'model_id'")

        device = "cuda" if "cuda" in self.device else "cpu"
        self._model = self._model.to(device)
        self._model.eval()

    def _unload_model(self) -> None:
        del self._model
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        audio_path = inputs.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = inputs.get("language", self._default_lang)
        lang_code = CONFORMER_LANG_MAP.get(language, language)

        transcriptions = self._model.transcribe([str(audio_path)])

        if isinstance(transcriptions, tuple):
            text = transcriptions[0][0] if transcriptions[0] else ""
        elif isinstance(transcriptions, list):
            text = transcriptions[0] if transcriptions else ""
        else:
            text = str(transcriptions)

        return {
            "text": text.strip(),
            "language": lang_code,
            "language_probability": 1.0,
            "duration": 0.0,
            "model": "indicconformer",
        }
