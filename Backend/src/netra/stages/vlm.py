from __future__ import annotations

from typing import Any

import torch

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry


@StageRegistry.register("vlm", variant="florence2")
class VisionLanguageStage(BaseStage):
    """
    Vision-Language fallback using Florence-2 or Moondream2.
    Used when OCR fails on handwritten or degraded documents.
    """

    def _load_model(self) -> None:
        from transformers import AutoProcessor
        try:
            from transformers import Florence2ForConditionalGeneration
            model_cls = Florence2ForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM

        model_id = self.config["model_id"]
        params = self.config.get("params", {})

        dtype_str = params.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16)

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=params.get("trust_remote_code", True),
        )
        self._model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=params.get("trust_remote_code", True),
        ).to(self.device)

        self._task_prompt = params.get("task_prompt", "<OCR>")

    def _unload_model(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image
        import numpy as np

        image = inputs.get("preprocessing_output", {}).get("image")
        image_path = inputs.get("image_path")

        if image is not None and isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif image_path:
            pil_image = Image.open(str(image_path)).convert("RGB")
        else:
            raise ValueError("No image provided for VLM")

        task_prompt = inputs.get("vlm_task", self._task_prompt)

        encoded = self._processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **encoded,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        post_processed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=pil_image.size,
        )

        text = post_processed.get(task_prompt, generated_text)
        if isinstance(text, dict):
            labels = text.get("labels", [])
            text = " ".join(labels) if labels else str(text)
        elif isinstance(text, list):
            text = " ".join(str(t) for t in text)

        return {
            "text": text,
            "lines": text.split("\n") if isinstance(text, str) else [str(text)],
            "num_lines": len(text.split("\n")) if isinstance(text, str) else 1,
            "source": "vlm_fallback",
        }


@StageRegistry.register("vlm", variant="moondream2")
class Moondream2Stage(BaseStage):
    """VLM using Moondream2 — more capable but heavier than Florence-2."""

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config["model_id"]
        params = self.config.get("params", {})
        dtype_str = params.get("torch_dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)

    def _unload_model(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image
        import numpy as np

        image = inputs.get("preprocessing_output", {}).get("image")
        image_path = inputs.get("image_path")

        if image is not None and isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
        elif image_path:
            pil_image = Image.open(str(image_path)).convert("RGB")
        else:
            raise ValueError("No image provided for VLM")

        prompt = inputs.get("vlm_prompt", "Extract all text from this document image.")
        enc_image = self._model.encode_image(pil_image)

        answer = self._model.answer_question(enc_image, prompt, self._tokenizer)

        return {
            "text": answer,
            "lines": answer.split("\n"),
            "num_lines": len(answer.split("\n")),
            "source": "moondream2",
        }


@StageRegistry.register("vlm", variant="ollama_vlm")
class OllamaVLMStage(BaseStage):
    """
    Vision-Language model via Ollama using MiniCPM-V 4.6 (1.3B, edge-optimized).
    Sends base64-encoded image to Ollama's chat API with image support.
    Optimized for PDF/document OCR tasks.
    """

    def _load_model(self) -> None:
        import urllib.request
        import json

        params = self.config.get("params", {})
        self._ollama_url = params.get("ollama_url", "http://localhost:11434")
        self._ollama_model = self.config.get("ollama_model", "openbmb/minicpm-v4.6")
        self._task_prompt = params.get("task_prompt", "Extract all text from this document image. Output only the extracted text.")

        try:
            req = urllib.request.Request(f"{self._ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"] for m in data.get("models", [])]
                if self._ollama_model not in available:
                    self.logger.warning(
                        "VLM model '%s' not in Ollama. Available: %s. Will pull on first use.",
                        self._ollama_model, available,
                    )
        except Exception as e:
            self.logger.warning("Ollama not reachable at %s: %s", self._ollama_url, e)

        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import base64
        import urllib.request
        import json
        from PIL import Image
        import numpy as np
        import io

        image = inputs.get("preprocessing_output", {}).get("image")
        image_path = inputs.get("image_path")

        if image is not None and isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif image_path:
            pil_image = Image.open(str(image_path))
        else:
            raise ValueError("No image provided for VLM")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = inputs.get("vlm_prompt", self._task_prompt)

        payload = json.dumps({
            "model": self._ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],
                },
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2048},
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
            raise RuntimeError(f"Ollama VLM failed: {e}") from e

        text = result.get("message", {}).get("content", "").strip()

        return {
            "text": text,
            "lines": text.split("\n") if text else [],
            "num_lines": len(text.split("\n")) if text else 0,
            "source": "ollama_vlm",
        }
