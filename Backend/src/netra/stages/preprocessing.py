from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry


@StageRegistry.register("preprocessing", variant="opencv")
@StageRegistry.register("preprocessing")
class PreprocessingStage(BaseStage):
    """
    Document image preprocessing using OpenCV.
    De-skew, binarize, enhance contrast — runs on CPU.
    """

    def __init__(self, stage_name: str, config: dict, device: str = "cpu") -> None:
        super().__init__(stage_name, config, device="cpu")
        self._params = config.get("params", {})

    def _load_model(self) -> None:
        pass

    def _unload_model(self) -> None:
        pass

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        image_path = inputs.get("image_path")
        image = inputs.get("image")

        if image is None and image_path:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

        if image is None:
            raise ValueError("No image or image_path provided")

        processed = self._pipeline(image)

        output_path = inputs.get("output_path")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)

        return {"image": processed, "original_shape": image.shape}

    def _pipeline(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        deskewed = self._deskew(denoised)
        enhanced = self._enhance_contrast(deskewed)

        # Prefer contrast-enhanced grayscale over hard binarization.
        # Modern OCR (EasyOCR / Paddle) reads enhanced text better than
        # adaptive-threshold binary masks, which often wipe out thin glyphs.
        use_binarize = bool(self._params.get("binarize", False))
        processed = self._adaptive_binarize(enhanced) if use_binarize else enhanced

        # OCR backends expect 3-channel BGR.
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        # Dark ink on light paper — use dark pixels as foreground.
        coords = np.column_stack(np.where(image < 128))
        if len(coords) < 5:
            coords = np.column_stack(np.where(image > 0))
        if len(coords) < 5:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _enhance_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def _adaptive_binarize(image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
