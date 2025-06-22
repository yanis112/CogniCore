"""
Free LLM Toolbox - A Python package for easy interaction with various LLMs and Vision models
"""

__version__ = "0.1.9"  # Mettez à jour la version!

__all__ = [
    "LanguageModel",
    "ImageAnalyzerAgent",
    "TextClassifier",
    "ImageClassifier",
    "InternetSearcher",
]

from .language_model import LanguageModel
from .vision_utils import ImageAnalyzerAgent
from .text_classifier_utils import TextClassifier
from .image_classifier_utils import ImageClassifier
from .internet_searcher_utils import InternetSearcher
# ... autres importations spécifiques ...
