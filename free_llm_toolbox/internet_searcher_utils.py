import os
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, Union, Generator
import yaml

load_dotenv()

class InternetSearcher:
    """
    A class for performing internet searches using language models with web-browsing capabilities.
    """

    def __init__(
        self,
        model_name: str = "compound-beta",
        provider: str = "groq",
        config: Optional[Union[dict, str]] = None,
    ):
        """
        Initializes the InternetSearcher.

        Args:
            model_name (str): The name of the model to use for searching. Defaults to "compound-beta".
            provider (str): The provider of the model. Currently only "groq" is supported.
            config (dict, str, or None): Configuration dictionary or yaml path.
        """
        config_dict = {}
        if config is not None:
            if isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif isinstance(config, dict):
                config_dict = config
            else:
                raise ValueError("config must be a dict, a yaml path (str), or None")

        # Explicit parameters override config
        self.model_name = config_dict.get("internet_search_model", model_name)
        self.provider = config_dict.get("internet_search_provider", provider)
        
        if self.provider.lower() != "groq":
            raise ValueError("Currently, only 'groq' provider is supported for InternetSearcher.")

        self.groq_token = os.getenv("GROQ_API_KEY")
        if not self.groq_token:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.groq_token)

        # Default generation parameters, can be overridden in search method
        self.default_params = {
            'temperature': 1.0,
            'max_tokens': 2048,
            'top_p': 1.0,
            'stop': None,
        }
        # Update defaults with config if present
        self.default_params['temperature'] = config_dict.get('temperature', self.default_params['temperature'])
        self.default_params['max_tokens'] = config_dict.get('max_tokens', self.default_params['max_tokens'])
        self.default_params['top_p'] = config_dict.get('top_p', self.default_params['top_p'])

    def search(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Performs an internet search based on the provided prompt.

        Args:
            prompt (str): The search query or prompt.
            stream (bool): Whether to stream the response. Defaults to False.
            **kwargs: Additional parameters to pass to the model completion call 
                      (e.g., temperature, max_tokens). These override defaults.

        Returns:
            Union[str, Generator[str, None, None]]: The search result as a string, 
                                                     or a generator if streaming.
        """
        params = self.default_params.copy()
        params.update(kwargs)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=params['temperature'],
            max_completion_tokens=params['max_tokens'],
            top_p=params['top_p'],
            stream=stream,
            stop=params['stop'],
        )

        if stream:
            def stream_generator():
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    yield content
            return stream_generator()
        else:
            return completion.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    searcher = InternetSearcher() # Can also be initialized with config="path/to/config.yaml"
    
    search_prompt = "What is the latest news about AI regulations in Europe?"
    
    print("--- Non-streaming search ---")
    result = searcher.search(search_prompt)
    print(result)

    print("\n--- Streaming search ---")
    stream_result = searcher.search(search_prompt, stream=True)
    for chunk in stream_result:
        print(chunk, end="", flush=True)
    print()
