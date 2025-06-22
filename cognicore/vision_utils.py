from PIL import Image
from groq import Groq
import google.generativeai as genai
import os
from typing import Optional, Union, List
import tempfile
import time
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
import base64
from dotenv import load_dotenv
from openai import OpenAI
import yaml

class ImageAnalyzerAgent:
    """
    A class for analyzing and describing images using various AI models.

    This class provides a unified interface to interact with different vision models,
    including Groq, Hugging Face's Florence-2, and Google's Gemini.

    Attributes:
        prompt (str): The default prompt used for image descriptions.
        device (str): The device to run models on, either 'cuda:0' or 'cpu'.
        torch_dtype (torch.dtype): The data type for torch tensors.
        token (str): API token for accessing the OpenAI service.
        endpoint (str): Endpoint URL for the OpenAI service.
        model_name (str): The name of the model to use for description if not precised in the method call.
        model (transformers.PreTrainedModel): The loaded model if using a florence2 based model.
        groq_token (str): API key for accessing the Groq service.
        processor (transformers.PreTrainedProcessor): The processor for preparing inputs for the model.
        groq_client (Groq): The client for accessing the Groq service.
        gemini_token (str): The API key for accessing the Google Gemini service.
    """

    prompt = "<MORE_DETAILED_CAPTION>"

    def __init__(self, vision_model: str = None, vision_provider: str = None, config=None):
        """
        Initializes the ImageAnalyzerAgent.
        Args:
            vision_model (str, optional): The name of the vision model to use.
            vision_provider (str, optional): The provider of the vision model.
            config (dict, str, or None): Configuration dictionary or yaml path containing generation parameters.
        """
        # Gestion de la config
        config_dict = {}
        if config is not None:
            if isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif isinstance(config, dict):
                config_dict = config
            else:
                raise ValueError("config must be a dict, a yaml path (str), or None")
        
        self.vision_model = vision_model or config_dict.get('vision_model')
        self.vision_provider = vision_provider or config_dict.get('vision_provider')

        self.config = {
            'temperature': config_dict.get('temperature', 1.0),
            'max_tokens': config_dict.get('max_tokens', 8000),
            'top_k': config_dict.get('top_k', 45),
            'top_p': config_dict.get('top_p', 0.95),
        }

        load_dotenv()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.token = os.getenv("GITHUB_TOKEN")
        self.endpoint = "https://models.inference.ai.azure.com"
        self.groq_token = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_token) if self.groq_token else None
        self.gemini_token = os.getenv("GOOGLE_API_KEY")
        if self.gemini_token:
            genai.configure(api_key=self.gemini_token)
        # Default refinement prompt template path
        self.default_refinement_prompt_path = "prompts/image_refinement_prompt.txt"

    def load_florence_model(self):
         """Load the Florence-2 model and processor."""
         self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
         self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
    
    def load_refinement_prompt(self, prompt_path: str) -> PromptTemplate:
        """Load and return the refinement prompt template."""
        with open(prompt_path, "r", encoding='utf-8') as f:
            template = f.read()
        return PromptTemplate.from_template(template)

    def get_image_data_url(self, image_file: str, image_format: str) -> str:
        """Convert an image file to a data URL string."""
        try:
            with open(image_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Could not read '{image_file}'.")
            exit()
        return f"data:image/{image_format};base64,{image_data}"

    def resize_image(self, image: Image, max_size: int = 512) -> Image:
        """Resize image if it exceeds maximum dimension while maintaining aspect ratio."""
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    
    def _describe_openai(self, image_path: str, prompt: str, model_name:str="gpt-4o-mini", max_size: int = 768, 
                        refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using OpenAI's model with optional refinement steps.
    
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the OpenAI model to use.
            max_size (int): Maximum size of the image dimension before resizing.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
    
        Returns:
            str: The generated image description text.
        """
        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        # Initial image processing
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.resize_image(image, max_size)
        
        # Save resized image to temporary file
        image_format = image_path.split('.')[-1]
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
            image.save(temp_file.name)
            image_data_url = self.get_image_data_url(temp_file.name, image_format)
        
        # Initial description
        current_description = self._get_openai_description(client, image_data_url, prompt, model_name)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                current_description = self._get_openai_description(client, image_data_url, refined_prompt, model_name)
        
        return current_description

    def _get_openai_description(self, client, image_data_url: str, prompt: str, model_name: str) -> str:
        """Helper method to get description from OpenAI."""
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes images in details.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "low"
                            },
                        },
                    ],
                },
            ],
            model=model_name,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        return response.choices[0].message.content

    def _describe_with_groq(self, image_path: Union[str, List[str]], prompt: str, model_name: str="meta-llama/llama-4-maverick-17b-128e-instruct", 
                           max_size: int = 768, refinement_steps: int = 1, 
                           refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe one or several images using Groq's vision model with optional refinement steps.
        Args:
            image_path (str | list[str]): Path or list of paths to image files.
            prompt (str): The prompt to guide the image description.
            model_name (str): The name of the Groq model to use.
            max_size (int): Maximum size of the image dimension before resizing.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
        Returns:
            str: The generated image description text.
        """
        # Image(s) processing
        if isinstance(image_path, str):
            paths_to_load = [image_path]
        elif isinstance(image_path, list):
            paths_to_load = image_path
        else:
            raise TypeError(f"Groq image_path must be a string or a list of strings, got {type(image_path)}")
        image_data_urls = []
        for path_item in paths_to_load:
            image = Image.open(path_item)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.resize_image(image, max_size)
            image_format = path_item.split('.')[-1]
            with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
                image.save(temp_file.name)
                image_data_url = self.get_image_data_url(temp_file.name, image_format)
            image_data_urls.append(image_data_url)
        # Compose message content
        content = [
            {"type": "text", "text": prompt}
        ] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
        ]
        # Initial description
        completion = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=self.config["temperature"],
            max_completion_tokens=self.config["max_tokens"],
            top_p=self.config["top_p"],
            stream=False,
            stop=None,
        )
        current_description = completion.choices[0].message.content
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                content = [
                    {"type": "text", "text": refined_prompt}
                ] + [
                    {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
                ]
                completion = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    temperature=self.config["temperature"],
                    max_completion_tokens=self.config["max_tokens"],
                    top_p=self.config["top_p"],
                    stream=False,
                    stop=None,
                )
                current_description = completion.choices[0].message.content
        return current_description

    def _describe_gemini(self, image_path: Union[str, List[str]], prompt: str, model_name: str="gemini-1.5-flash",
                        refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describes an image ou une liste d'images avec Gemini.
        Args:
            image_path (str | list[str]): Chemin ou liste de chemins d'images.
            prompt (str): Prompt pour la description.
            model_name (str): Nom du modèle Gemini.
            refinement_steps (int): Nombre d'itérations de raffinement.
            refinement_prompt_path (str, optional): Chemin du prompt de raffinement.
        Returns:
            str: Description générée.
        """
        try:
            generation_config = {
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "top_k": self.config["top_k"],
                "max_output_tokens": self.config["max_tokens"],
                "response_mime_type": "text/plain",
            }
            pil_images = []
            if isinstance(image_path, str):
                paths_to_load = [image_path]
            elif isinstance(image_path, list):
                paths_to_load = image_path
            else:
                raise TypeError(f"Gemini image_path must be a string or a list of strings, got {type(image_path)}")
            for path_item in paths_to_load:
                image = Image.open(path_item)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                pil_images.append(image)
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            # Initial description: images d'abord, puis prompt
            initial_content_parts = pil_images + [prompt]
            current_description = model.generate_content(initial_content_parts).text
            # Raffinement
            if refinement_steps > 1:
                template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
                for _ in range(refinement_steps - 1):
                    refined_prompt_text = template.format(
                        previous_description=current_description
                    )
                    refined_content_parts = pil_images + [refined_prompt_text]
                    current_description = model.generate_content(refined_content_parts).text
            return current_description
        except Exception as e:
            return f"Error during Gemini description: {e}"

    def _describe_with_florence(self, image_path: str, prompt: str, 
                              refinement_steps: int = 1, refinement_prompt_path: Optional[str] = None) -> str:
        """
        Describe an image using the florence2 model with optional refinement steps.
        
        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to guide the image description.
            refinement_steps (int): Number of refinement iterations. Default is 1 (no refinement).
            refinement_prompt_path (str, optional): Path to the refinement prompt template.
        Returns:
            str: The generated image description text.
        """
        self.load_florence_model()
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Initial description
        current_description = self._get_florence_description(image, prompt)
        
        # Refinement steps
        if refinement_steps > 1:
            template = self.load_refinement_prompt(refinement_prompt_path or self.default_refinement_prompt_path)
            
            for _ in range(refinement_steps - 1):
                refined_prompt = template.format(
                    original_query=prompt,
                    previous_answer=current_description
                )
                current_description = self._get_florence_description(image, refined_prompt)
        
        return current_description

    def _get_florence_description(self, image: Image, prompt: str) -> str:
        """Helper method to get description from Florence model."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.config["max_tokens"],
            num_beams=3,
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"]
        )
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        caption = self.processor.post_process_generation(caption, task="<CAPTION>", image_size=(image.width, image.height))
        return caption["<CAPTION>"]
    
    def describe(self, image_path: Union[str, List[str]], prompt: Optional[str] = None, vision_provider: str = "florence2", 
                vision_model: Optional[str]=None, refinement_steps: int = 1, 
                refinement_prompt_path: Optional[str] = None) -> str:
        """
        Unified method to describe an image or images using a specified provider.
        Args:
            image_path (str | list[str]): Chemin ou liste de chemins d'images (liste supportée pour Gemini et Groq).
            prompt (str, optional): Prompt pour la description.
            vision_provider (str, optional): Provider du modèle.
            vision_model (str, optional): Nom du modèle.
            refinement_steps (int): Nombre d'itérations de raffinement.
            refinement_prompt_path (str, optional): Chemin du prompt de raffinement.
        Returns:
            str: Description générée.
        Raises:
            ValueError: Provider non supporté.
            TypeError: Si image_path est une liste pour un provider qui ne le supporte pas.
        """
        if prompt is None:
            prompt = self.prompt
        
        # Use the class's default model and provider if not specified in the call
        vision_provider = vision_provider or self.vision_provider
        vision_model = vision_model or self.vision_model

        if vision_provider == "florence2":
            if not isinstance(image_path, str):
                raise TypeError(f"Provider '{vision_provider}' currently supports only a single image path (str), but got type {type(image_path)}.")
            return self._describe_with_florence(image_path, prompt, refinement_steps, refinement_prompt_path)
        elif vision_provider == "groq":
            return self._describe_with_groq(
                image_path, prompt, 
                model_name=vision_model if vision_model else "meta-llama/llama-4-maverick-17b-128e-instruct",
                refinement_steps=refinement_steps,
                refinement_prompt_path=refinement_prompt_path
            )
        elif vision_provider == "github":
            if not isinstance(image_path, str):
                raise TypeError(f"Provider '{vision_provider}' currently supports only a single image path (str), but got type {type(image_path)}.")
            return self._describe_openai(
                image_path, prompt, 
                model_name=vision_model if vision_model else 'gpt-4o-mini',
                refinement_steps=refinement_steps,
                refinement_prompt_path=refinement_prompt_path
            )
        elif vision_provider == "gemini":
            return self._describe_gemini(
                image_path, prompt, 
                model_name=vision_model if vision_model else "gemini-1.5-flash",
                refinement_steps=refinement_steps,
                refinement_prompt_path=refinement_prompt_path
            )
        else:
            raise ValueError(
                "Unsupported provider. Choose from 'florence2', 'groq', 'github', or 'gemini'."
            )

# Example usage
if __name__ == "__main__":
    config_path = "exemple_config.yaml"
    analyzer = ImageAnalyzerAgent(config=config_path)
    image_path1 = "test_data/borsalino-rostro-min.webp"
    image_path2 = "test_data/borsalino-rostro-min.webp"  # pour test multi-image, tu peux changer le chemin
    prompt = "Describe precisely the content of the image(s)."

    # Test single image Groq
    description_groq = analyzer.describe(
        image_path1, prompt=prompt, vision_provider="groq"
    )
    print(f"Groq (single image) Description:\n{description_groq}")
    print("#############################################")

    # Test multi-image Groq
    description_groq_multi = analyzer.describe(
        [image_path1, image_path2], prompt="Describe both images", vision_provider="groq"
    )
    print(f"Groq (multi-image) Description:\n{description_groq_multi}")
    print("#############################################")

    # Test single image Gemini
    description_gemini = analyzer.describe(
        image_path1, prompt=prompt, vision_provider="gemini"
    )
    print(f"\nGemini (single image) Description:\n{description_gemini}")
    print("#############################################")

    # Test multi-image Gemini
    description_gemini_multi = analyzer.describe(
        [image_path1, image_path2], prompt="Describe both images", vision_provider="gemini"
    )
    print(f"\nGemini (multi-image) Description:\n{description_gemini_multi}")
    print("#############################################")

    # Test single image github
    description_github = analyzer.describe(
        image_path1, prompt=prompt, vision_provider="github"
    )
    print(f"\nGithub (GPT-4o-mini) Description:\n{description_github}")
    print("#############################################")

