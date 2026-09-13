"""
Pipeline stages — each module auto-registers with the StageRegistry.
Import this package to make all stages discoverable.
"""

from netra.stages.preprocessing import PreprocessingStage  # noqa: F401
from netra.stages.ocr import PaddleOCRStage, EasyOCRStage, SuryaOCRStage  # noqa: F401
from netra.stages.asr import WhisperASRStage, IndicConformerASRStage  # noqa: F401
from netra.stages.llm import HuggingFaceLLMStage, OllamaLLMStage, OLLMStage  # noqa: F401
from netra.stages.nmt import IndicTransNMTStage, NLLBTranslationStage, OllamaNMTStage  # noqa: F401
from netra.stages.tts import IndicTTSStage, PiperTTSStage, EdgeTTSStage, IndicF5TTSStage  # noqa: F401
from netra.stages.vlm import VisionLanguageStage, Moondream2Stage, OllamaVLMStage  # noqa: F401
from netra.stages.embedding import EmbeddingStage, OllamaEmbeddingStage  # noqa: F401
from netra.stages.retrieval import FAISSRetrievalStage, OllamaRetrievalStage  # noqa: F401
from netra.stages.knowledge_graph import GraphKBStage  # noqa: F401
