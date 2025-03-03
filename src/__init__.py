"""
Free LLM Toolbox - A Python package for easy interaction with various LLMs and Vision models
"""
from .vision_utils import ImageAnalyzerAgent
from .language_model import LanguageModel

__version__ = "0.1.0"

__all__ = [
    "LanguageModel",
    "ImageAnalyzerAgent",
]