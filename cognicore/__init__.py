"""
Free LLM Toolbox - A Python package for easy interaction with various LLMs and Vision models
"""

# La version est maintenant gérée dynamiquement par hatch-vcs.
# Ce code tente d'importer la version depuis le fichier __version__.py,
# qui est généré automatiquement lors de la construction du package.
try:
    from .__version__ import __version__
except ImportError:
    # En cas d'échec (par exemple, si le package n'est pas installé),
    # on définit une version par défaut.
    __version__ = "0.0.0"

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
