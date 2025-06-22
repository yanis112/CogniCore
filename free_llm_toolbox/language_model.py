import os
from functools import lru_cache
from typing import List, Union, Callable, Dict, Any, Optional, Generator, Type
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools.structured import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import StrOutputParser
import yaml

# Import all providers at the top level
import subprocess
from langchain_groq import ChatGroq
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama
from langchain_cerebras import ChatCerebras
import ollama

# Load environment variables
load_dotenv()

class LanguageModel:
    """
    A unified interface for interacting with various Language Models (LLMs).
    This class handles model initialization, caching, and provides methods for text generation
    with support for tools, JSON formatting, and streaming.

    Attributes:
        model_name (str): The name of the language model to use
        provider (str): The provider of the language model (e.g., "groq", "google", "github")
        temperature (float): Controls randomness in text generation (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        top_k (int): Number of tokens to consider for sampling
        top_p (float): Cumulative probability threshold for sampling
        system_prompt (str): System-level prompt to guide model behavior
        
    Example:
        ```python
        session = LanguageModel(
            model_name="llama-3-70b",
            provider="groq",
            temperature=0.7
        )
        
        response = session.answer("What is quantum computing?")
        ```
    """
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = None,
        temperature: float = 1.0,
        max_tokens: int = 20000,
        top_k: int = 45,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
        config: Optional[Union[dict, str]] = None
    ):
        # Gestion de la config
        if config is not None:
            if isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif isinstance(config, dict):
                config_dict = config
            else:
                raise ValueError("config must be a dict, a yaml path (str), or None")
            # On surcharge les arguments
            model_name = config_dict.get('model_name', model_name)
            provider = config_dict.get('llm_provider', provider)
            temperature = config_dict.get('temperature', temperature)
            max_tokens = config_dict.get('max_tokens', max_tokens)
            top_k = config_dict.get('top_k', top_k)
            top_p = config_dict.get('top_p', top_p)
            system_prompt = config_dict.get('system_prompt', system_prompt)
        if model_name is None or provider is None:
            raise ValueError("model_name and provider must be specified, either directly or via config")
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.context_window_size = 0
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the language model based on the specified provider."""
        
        if self.provider == "groq":
            self.context_window_size = 131072
            self.chat_model = ChatGroq(
                temperature=self.temperature,
                model_name=self.model_name,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                max_tokens=self.max_tokens,
            )

        elif self.provider == "sambanova":
            os.environ["SAMBANOVA_API_KEY"] = os.getenv("SAMBANOVA_API_KEY")
            self.context_window_size = 8000
            self.chat_model = ChatSambaNovaCloud(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        elif self.provider == "github":
            from src.aux_utils.github_llm import GithubLLM
            self.chat_model = GithubLLM(
                github_token=os.getenv("GITHUB_TOKEN"),
                model_name=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            )

        elif self.provider == "google":
            self.chat_model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=None,
                max_retries=2,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                top_k=self.top_k,
                top_p=self.top_p,
            )

        elif self.provider == "ollama":
            self._ensure_model_pulled()
            template = ollama.show(self.model_name)["template"]
            self.context_window_size = 8192
            self.chat_model = ChatOllama(
                model=self.model_name,
                keep_alive=0,
                num_ctx=self.context_window_size,
                temperature=self.temperature,
                template=template,
            )

        elif self.provider == "cerebras":
            os.environ["CEREBRAS_API_KEY"] = os.getenv("CEREBRAS_API_KEY")
            self.chat_model = ChatCerebras(
                model=self.model_name,
            )

        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Please choose from: groq, sambanova, github, google, ollama, cerebras"
            )

        self.chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{text}")
        ])

    def _ensure_model_pulled(self) -> None:
        """Ensures that an Ollama model is pulled before use."""
        try:
            subprocess.run(
                ["ollama", "pull", self.model_name],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull Ollama model: {e}")

    def bind_tools(self, tool_list: List[Union[StructuredTool, Callable, Dict[str, Any]]]) -> 'LanguageModel':
        """
        Binds tools to the language model for function calling capabilities.
        
        Args:
            tool_list: List of tools to bind to the model
            
        Returns:
            A new LanguageModel instance with the tools bound
        """
        if not tool_list:
            return self

        if self.provider == "groq":
            formatted_tools = [
                convert_to_openai_function(tool.func if isinstance(tool, StructuredTool) else tool)
                for tool in tool_list
            ]
            new_chat_model = self.chat_model.bind_tools(formatted_tools)
        else:
            new_chat_model = self.chat_model.bind_tools(tool_list)

        new_session = LanguageModel(
            model_name=self.model_name,
            provider=self.provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            system_prompt=self.system_prompt
        )
        new_session.chat_model = new_chat_model
        new_session.chat_prompt_template = self.chat_prompt_template
        return new_session

    def answer(
        self,
        prompt: str,
        prompt_file: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        json_formatting: bool = False,
        pydantic_object: Optional[Type[BaseModel]] = None,
        format_type: Optional[str] = None,
        tool_list: List[Any] = [],
        stream: bool = False,
    ) -> Union[str, Generator, tuple]:
        """
        Generate a response using the language model.

        Args:
            prompt: The input prompt for the model
            prompt_file: Optional path to a file containing a prompt template
            prompt_variables: Variables to format the prompt template with
            json_formatting: Whether to format the output as JSON
            pydantic_object: Pydantic model for JSON validation
            format_type: Type of structured output format
            tool_list: List of tools for the model to use
            stream: Whether to stream the response
            
        Returns:
            Generated response as string, generator (if streaming), or tuple (if using tools)
        """
        if prompt_file:
            with open(prompt_file) as f:
                prompt_template = PromptTemplate.from_template(f.read())
            prompt = prompt_template.format(**(prompt_variables or {}))

        if tool_list:
            session_with_tools = self.bind_tools(tool_list)
            chain = session_with_tools.chat_prompt_template | session_with_tools.chat_model
            response = chain.invoke({"text": prompt})
            return response.content, response.tool_calls

        if json_formatting and issubclass(pydantic_object, BaseModel):
            parser = JsonOutputParser(pydantic_object=pydantic_object)
            format_instructions = parser.get_format_instructions()
            
            if format_type:
                from src.main_utils.utils import get_strutured_format
                schema = parser._get_schema(pydantic_object)
                format_instructions = get_strutured_format(format_type) + "```" + str(schema) + "```"

            prompt = f"Answer the user query.\n{prompt}\n{format_instructions}"
            chain = self.chat_prompt_template | self.chat_model | parser
            try:
                return chain.invoke({"text": prompt})
            except Exception as e:
                raise ValueError(f"JSON parsing error: {e}")

        chain = self.chat_prompt_template | self.chat_model | StrOutputParser()
        
        if stream:
            return chain.stream({"text": prompt})
        return chain.invoke({"text": prompt})

if __name__ == "__main__":
    # Test simple avec config dict
    config = {
        'model_name': 'meta-llama/llama-4-maverick-17b-128e-instruct',
        'llm_provider': 'groq',
        'temperature': 0.7,
        'max_tokens': 128,
        'top_k': 20,
        'top_p': 0.9,
        'system_prompt': 'You are a test assistant.'
    }
    llm = LanguageModel(config=config)
    print(llm.answer("Quel est le sens de la vie ?"))
  