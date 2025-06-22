from .vision_utils import ImageAnalyzerAgent
from langchain_core.prompts import PromptTemplate
import os
from typing import Dict, Any, Optional, Union
import yaml

class ImageClassifier(ImageAnalyzerAgent):
    def __init__(
        self,
        labels_dict: Optional[Dict[int, Dict[str, str]]] = None,
        classifier_system_prompt: Optional[str] = None,
        image_classification_model: Optional[str] = None,
        image_classification_provider: Optional[str] = None,
        config: Optional[Union[dict, str]] = None,
    ):
        config_dict = {}
        if config is not None:
            if isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif isinstance(config, dict):
                config_dict = config
            else:
                raise ValueError("config must be a dict, a yaml path (str), or None")
        # Les paramètres explicites sont prioritaires sur la config
        labels_dict = labels_dict if labels_dict is not None else config_dict.get("image_classification_labels_dict")
        classifier_system_prompt = classifier_system_prompt if classifier_system_prompt is not None else config_dict.get("image_classifier_system_prompt")
        image_classification_model = image_classification_model if image_classification_model is not None else config_dict.get("image_classification_model")
        image_classification_provider = image_classification_provider if image_classification_provider is not None else config_dict.get("image_classification_provider")
        if labels_dict is None:
            raise ValueError("labels_dict must be provided, either directly or via config")
        self.labels_dict = labels_dict
        self.class_names = [v["class_name"] for k, v in sorted(labels_dict.items())]
        self.class_indices = list(sorted(labels_dict.keys()))
        self.class_descriptions = [v["description"] for k, v in sorted(labels_dict.items())]
        self.classifier_system_prompt = classifier_system_prompt or "You are an agent in charge of classifying images into different categories."
        self.image_classification_model = image_classification_model
        self.image_classification_provider = image_classification_provider
        # Appel du super constructeur avec le bon modèle/config
        super().__init__(config=config)
        # Chargement des prompts
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts_dir = os.path.join(base_dir, "prompts")
        self.prompt_path_index = os.path.join(self.prompts_dir, "llm_image_classification_index.txt")
        self.prompt_path_name = os.path.join(self.prompts_dir, "llm_image_classification_name.txt")
        with open(self.prompt_path_index, "r", encoding="utf-8") as f:
            self.prompt_template_index = PromptTemplate.from_template(f.read())
        with open(self.prompt_path_name, "r", encoding="utf-8") as f:
            self.prompt_template_name = PromptTemplate.from_template(f.read())

    def classify(self, image_path: str, return_class_name: bool = False) -> Union[int, str]:
        """
        Classifie l'image parmi les classes fournies. Retourne l'index (par défaut) ou le nom de la classe.
        """
        labels_dict_str = str(self.labels_dict)
        if return_class_name:
            prompt = self.prompt_template_name.format(labels_dict=labels_dict_str)
        else:
            prompt = self.prompt_template_index.format(labels_dict=labels_dict_str)
        # Utilise la méthode describe héritée pour obtenir une description de l'image
        # puis on pose la question de classification à l'IA
        image_description = self.describe(image_path, prompt=prompt, vllm_provider=self.image_classification_provider, vllm_name=self.image_classification_model)
        response_str = str(image_description).strip()
        if return_class_name:
            return response_str
        try:
            return int(response_str)
        except Exception:
            raise ValueError(f"Model did not return a valid class index: {response_str}")

if __name__ == "__main__":
    # Exemple d'utilisation avec écrasement du dictionnaire de classes
    config_path = "exemple_config.yaml"
    labels_dict = {
        0: {"class_name": "animal", "description": "Image contenant un animal."},
        1: {"class_name": "ville", "description": "Image représentant une ville ou un lieu urbain."},
        2: {"class_name": "autre", "description": "Tout autre type d'image."}
    }
    classifier = ImageClassifier(config=config_path, labels_dict=labels_dict)
    test_image = "test_data/borsalino-rostro-min.webp"
    print("Image :", test_image)
    print("Index de classe prédit :", classifier.classify(test_image))
    print("Nom de classe prédit :", classifier.classify(test_image, return_class_name=True))