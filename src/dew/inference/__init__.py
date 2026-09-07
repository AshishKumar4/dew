"""Inference tasks: native generation bound to weights, and external engines."""

from .clients import Completion, OllamaCompletion, OpenAICompletion
from .tasks import BlockGeneration, Processor, TextGeneration

__all__ = ["BlockGeneration", "Completion", "OllamaCompletion", "OpenAICompletion", "Processor",
           "TextGeneration"]
