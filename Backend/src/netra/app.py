from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from netra.config.settings import Settings
from netra.core.pipeline import Pipeline
from netra.utils.device import get_device_info
from netra.utils.log import setup_logging


class NetraApp:
    """Main application class for the NETRA pipeline."""

    def __init__(self, config_dir: str | None = None) -> None:
        self.settings = Settings(config_dir)
        setup_logging(
            level=self.settings.logging.level,
            log_format=self.settings.logging.format,
            log_file=self.settings.logging.file,
        )

        import netra.stages  # noqa: F401 — triggers stage registration

        self.pipeline = Pipeline(self.settings)

    def process_document(
        self,
        image_path: str,
        target_language: str | None = None,
        task: str = "simplify",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Full pipeline: preprocess -> OCR -> LLM -> NMT -> TTS."""
        target_lang = target_language or self.settings.languages.default_target
        out_dir = output_dir or self.settings.paths.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        prompt_key = {
            "classify": "document_classification",
            "simplify": "document_simplification",
            "qa": "document_qa",
        }.get(task, "document_simplification")

        prompt_config = self.settings.get_prompt(prompt_key)

        inputs: dict[str, Any] = {
            "image_path": image_path,
            "target_language": target_lang,
            "system_prompt": prompt_config["system"],
            "user_prompt": prompt_config["user_template"],
            "output_dir": out_dir,
        }

        results = self.pipeline.run(inputs)
        return self._format_output(results)

    def process_voice(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """ASR pipeline: transcribe voice command."""
        inputs: dict[str, Any] = {
            "audio_path": audio_path,
            "language": language,
        }
        result = self.pipeline.run_stage("asr", inputs)
        return {"asr": result.data if result.success else result.error}

    def translate_text(
        self,
        text: str,
        source_language: str = "en",
        target_language: str = "hi",
    ) -> dict[str, Any]:
        """Standalone translation."""
        inputs = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        result = self.pipeline.run_stage("nmt", inputs)
        return result.data if result.success else {"error": result.error}

    def get_system_info(self) -> dict[str, Any]:
        device = get_device_info()
        return {
            "device": {
                "platform": device.platform,
                "gpu": device.gpu_name,
                "vram_total_mb": device.gpu_memory_total_mb,
                "vram_free_mb": device.gpu_memory_free_mb,
                "cuda": device.cuda_version,
                "is_jetson": device.is_jetson,
                "jetson_model": device.jetson_model,
                "tensorrt": device.tensorrt_available,
            },
            "pipeline": self.pipeline.list_stages(),
            "memory": self.pipeline.get_memory_status(),
        }

    def shutdown(self) -> None:
        self.pipeline.shutdown()

    @staticmethod
    def _format_output(results: dict) -> dict[str, Any]:
        output: dict[str, Any] = {"success": True, "stages": {}}

        for stage_name, result in results.items():
            output["stages"][stage_name] = {
                "success": result.success,
                "latency_ms": result.latency_ms,
                "data": result.data,
                "error": result.error,
            }
            if not result.success:
                output["success"] = False

        return output


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="netra",
        description="NETRA — Open-Source INT4 LLM Pipeline for Indian Document Processing",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config directory",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    doc_parser = subparsers.add_parser("process", help="Process a document image")
    doc_parser.add_argument("image", help="Path to document image")
    doc_parser.add_argument("--lang", "-l", default=None, help="Target language code (e.g. hi, ta, te)")
    doc_parser.add_argument("--task", "-t", default="simplify", choices=["classify", "simplify", "qa"])
    doc_parser.add_argument("--output", "-o", default=None, help="Output directory")

    voice_parser = subparsers.add_parser("voice", help="Process voice command")
    voice_parser.add_argument("audio", help="Path to audio file")
    voice_parser.add_argument("--lang", "-l", default=None, help="Expected language")

    translate_parser = subparsers.add_parser("translate", help="Translate text")
    translate_parser.add_argument("text", help="Text to translate")
    translate_parser.add_argument("--src", "-s", default="en", help="Source language")
    translate_parser.add_argument("--tgt", "-t", default="hi", help="Target language")

    subparsers.add_parser("info", help="Show system and pipeline info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    app = NetraApp(config_dir=args.config)

    try:
        if args.command == "process":
            result = app.process_document(
                image_path=args.image,
                target_language=args.lang,
                task=args.task,
                output_dir=args.output,
            )
        elif args.command == "voice":
            result = app.process_voice(audio_path=args.audio, language=args.lang)
        elif args.command == "translate":
            result = app.translate_text(
                text=args.text,
                source_language=args.src,
                target_language=args.tgt,
            )
        elif args.command == "info":
            result = app.get_system_info()
        else:
            parser.print_help()
            sys.exit(1)

        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
