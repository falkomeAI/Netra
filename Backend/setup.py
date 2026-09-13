from setuptools import setup, find_packages

setup(
    name="netra",
    version="2.0.0",
    description="NETRA v2 — RAG + GraphKB + BitNet LLM Pipeline for Indian Document Processing",
    author="NETRA Team",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "torch>=2.1",
        "transformers>=4.40",
        "huggingface-hub>=0.20",
    ],
    extras_require={
        "rag": [
            "sentence-transformers>=3.0",
            "faiss-cpu>=1.8",
        ],
        "graph": [
            "networkx>=3.2",
        ],
        "ocr": [
            "paddleocr>=2.7",
            "paddlepaddle-gpu>=2.6",
            "easyocr>=1.7",
        ],
        "ocr-surya": [
            "surya-ocr>=0.4",
        ],
        "asr": [
            "faster-whisper>=1.0",
        ],
        "tts": [
            "TTS>=0.22",
            "edge-tts>=6.1",
        ],
        "tts-piper": [
            "piper-tts>=1.2",
        ],
        "quantization": [
            "autoawq>=0.2",
            "auto-gptq>=0.7",
            "optimum>=1.17",
        ],
        "tensorrt": [
            "tensorrt>=10.0",
        ],
        "all": [
            "sentence-transformers>=3.0",
            "faiss-cpu>=1.8",
            "networkx>=3.2",
            "paddleocr>=2.7",
            "paddlepaddle-gpu>=2.6",
            "easyocr>=1.7",
            "faster-whisper>=1.0",
            "TTS>=0.22",
            "edge-tts>=6.1",
            "autoawq>=0.2",
            "auto-gptq>=0.7",
            "optimum>=1.17",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "netra=netra.app:main",
        ],
    },
)
