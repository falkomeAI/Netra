from __future__ import annotations

import os
from typing import Any

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry


@StageRegistry.register("ocr", variant="paddleocr")
class PaddleOCRStage(BaseStage):
    """OCR using PaddleOCR v4/v5 — supports all Indic scripts."""

    def _load_model(self) -> None:
        # Force-disable oneDNN/MKLDNN — some Paddle CPU builds crash otherwise.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_onednn"] = "0"
        from paddleocr import PaddleOCR

        params = self.config.get("params", {})
        lang = params.get("lang", "en")

        init_kwargs = {
            "lang": lang,
        }

        # PaddleOCR 3.x compatible: try new API params, fall back gracefully
        try:
            self._model = PaddleOCR(**init_kwargs)
        except Exception:
            self._model = PaddleOCR(lang=lang)

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        image = inputs.get("preprocessing_output", {}).get("image")
        if image is None:
            image = inputs.get("image")
        if image is None:
            image_path = inputs.get("image_path")
            if image_path is None:
                raise ValueError("No image data provided for OCR")
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

        if isinstance(image, np.ndarray):
            import cv2

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.ascontiguousarray(image, dtype=np.uint8)

        # Try PaddleOCR 3.x predict() first, fall back to 2.x ocr()
        try:
            result = self._model.predict(image)
        except (AttributeError, TypeError):
            result = self._model.ocr(image, cls=True)

        text_lines = []
        boxes = []
        confidences = []

        def _consume_ocr_item(item) -> None:
            # PaddleOCR 3.x result objects
            rec_texts = getattr(item, "rec_texts", None)
            if rec_texts:
                rec_scores = getattr(item, "rec_scores", None)
                dt_polys = getattr(item, "dt_polys", None)
                for i, txt in enumerate(rec_texts):
                    text_lines.append(str(txt))
                    if rec_scores and i < len(rec_scores):
                        confidences.append(float(rec_scores[i]))
                    if dt_polys and i < len(dt_polys):
                        boxes.append(dt_polys[i])
                return
            # PaddleOCR 2.x page list: [[box, (text, conf)], ...]
            if isinstance(item, list):
                for line in item:
                    if isinstance(line, (list, tuple)) and len(line) == 2:
                        box_data, text_data = line
                        if isinstance(text_data, (list, tuple)) and len(text_data) == 2:
                            text, conf = text_data
                            text_lines.append(str(text))
                            boxes.append(box_data)
                            confidences.append(float(conf))

        if result is None:
            pass
        elif isinstance(result, list):
            for page in result:
                if not page:
                    continue
                _consume_ocr_item(page)
        else:
            _consume_ocr_item(result)

        if not text_lines:
            raise ValueError("OCR produced no text — document may be blank or unreadable")

        return {
            "text": "\n".join(text_lines),
            "lines": text_lines,
            "boxes": boxes,
            "confidences": confidences,
            "num_lines": len(text_lines),
        }


@StageRegistry.register("ocr", variant="easyocr")
class EasyOCRStage(BaseStage):
    """OCR using EasyOCR — reliable multi-script OCR supporting Devanagari, Tamil, Telugu, Bengali, Kannada, etc."""

    LANG_MAP = {
        "hi": "hi", "mr": "mr", "ne": "hi", "sa": "hi", "mai": "hi",
        "kok": "hi", "doi": "hi", "brx": "hi",
        "ta": "ta", "te": "te", "bn": "bn", "as": "bn",
        "kn": "kn", "ml": "ml", "gu": "gu", "pa": "pa",
        "ur": "ur", "sd": "ur", "ks": "ur",
        "en": "en",
    }

    def _load_model(self) -> None:
        import easyocr
        params = self.config.get("params", {})
        languages = params.get("languages", ["en", "hi"])
        gpu = "cuda" in self.device
        self._languages = languages
        self._gpu = gpu
        self._model = easyocr.Reader(languages, gpu=gpu)

    def _unload_model(self) -> None:
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        # Prefer original camera/upload path — EasyOCR has its own preprocessing
        # and often reads raw images better than OpenCV deskew/CLAHE output.
        image_path = inputs.get("image_path")
        image = None
        if image_path:
            image = str(image_path)
        else:
            image = inputs.get("preprocessing_output", {}).get("image")
            if image is None:
                image = inputs.get("image")
            if image is None:
                raise ValueError("No image data provided for OCR")
            if isinstance(image, np.ndarray):
                import cv2

                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = np.ascontiguousarray(image, dtype=np.uint8)

        target_lang = inputs.get("target_language", "hi")
        needed_lang = self.LANG_MAP.get(target_lang, "en")
        if needed_lang not in self._languages:
            try:
                import easyocr

                old_model = self._model
                new_langs = list(set(self._languages + [needed_lang]))
                self._model = easyocr.Reader(new_langs, gpu=self._gpu)
                self._languages = new_langs
                del old_model
            except Exception as e:
                self.logger.warning("Failed to reload EasyOCR with lang '%s': %s", needed_lang, e)

        results = self._model.readtext(image)

        text_lines = []
        boxes = []
        confidences = []

        for box, text, conf in results:
            text_lines.append(str(text))
            # Convert numpy types so Flask jsonify can serialize the response.
            if hasattr(box, "tolist"):
                boxes.append(box.tolist())
            else:
                boxes.append([[float(x), float(y)] for x, y in box])
            confidences.append(float(conf))

        if not text_lines:
            raise ValueError("OCR produced no text — document may be blank or unreadable")

        return {
            "text": "\n".join(text_lines),
            "lines": text_lines,
            "boxes": boxes,
            "confidences": confidences,
            "num_lines": len(text_lines),
        }


@StageRegistry.register("ocr", variant="surya")
class SuryaOCRStage(BaseStage):
    """OCR using Surya — transformer-based, high accuracy for Indic scripts and handwritten docs."""

    def _load_model(self) -> None:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        self._det_predictor = DetectionPredictor()
        self._rec_predictor = RecognitionPredictor()
        self._model = True

    def _unload_model(self) -> None:
        self._det_predictor = None
        self._rec_predictor = None
        self._model = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image
        import numpy as np

        image = inputs.get("preprocessing_output", {}).get("image")
        image_path = inputs.get("image_path")

        if image is not None:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
        elif image_path:
            pil_image = Image.open(str(image_path))
        else:
            raise ValueError("No image data provided for OCR")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        target_lang = inputs.get("target_language", "en")
        languages = inputs.get("languages", [target_lang, "en"])
        if isinstance(languages, str):
            languages = [languages]
        if "en" not in languages:
            languages.append("en")

        # Surya >= 0.4 API: use predictor objects directly
        det_result = self._det_predictor([pil_image])
        rec_result = self._rec_predictor([pil_image], det_result)

        text_lines = []
        confidences = []

        if rec_result:
            page = rec_result[0]
            # Handle both old (.text_lines) and new (.blocks) API
            if hasattr(page, "text_lines"):
                for line in page.text_lines:
                    text_lines.append(line.text)
                    confidences.append(getattr(line, "confidence", 1.0))
            elif hasattr(page, "blocks"):
                for block in page.blocks:
                    for line in getattr(block, "lines", []):
                        for span in getattr(line, "spans", []):
                            text_lines.append(span.text)
                            confidences.append(getattr(span, "confidence", 1.0))

        return {
            "text": "\n".join(text_lines),
            "lines": text_lines,
            "confidences": confidences,
            "num_lines": len(text_lines),
        }
