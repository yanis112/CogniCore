# CogniCore 🚀

A Python package that provides a unified, easy-to-use interface for working with various Language Models (LLMs) and Vision Models from multiple providers. 🎯 It focuses on leveraging the generous free tiers offered by AI platforms.

## Features

- Text generation with multiple LLM providers support
- Image analysis and description capabilities
- Support for models like Llama, Groq, and Google's Gemini
- Streaming responses
- Tool integration support
- JSON output formatting
- Customizable system prompts

## Installation 💻

```bash
uv pip install cognicore
```

## Configuration ⚙️

Before using the library, you need to configure your API keys in a `.env` file:

```env
GROQ_API_KEY=your_groq_key
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_key
SAMBANOVA_API_KEY=your_sambanova_key
CEREBRAS_API_KEY=your_cerebras_key
```

## Quick Start

### Text Generation

```python
from cognicore import LanguageModel

# Initialize a session with your preferred model
session = LanguageModel(
    model_name="gemini-2.0-flash",
    provider="google",
    temperature=0.7
)

# Generate a response
response = session.answer("What is the capital of France?")
print(response)
```

### Image Analysis

```python
from cognicore import ImageAnalyzerAgent

analyzer = ImageAnalyzerAgent()
description = analyzer.describe(
    "path/to/image.jpg",
    prompt="Describe the image",
    vllm_provider="groq",
    vllm_name="llama-3.2-90b-vision-preview"
)
print(description)
```

## Usage 🎮

### Text Models 📚

```python
from cognicore import LanguageModel

# Initialize a session with your preferred model
session = LanguageModel(
    model_name="llama-3-70b",
    provider="groq",
    temperature=0.7,
    top_k=45,
    top_p=0.95
)

# Simple text generation
response = session.answer("What is the capital of France?")

# JSON-formatted response with Pydantic validation
from pydantic import BaseModel

class LocationInfo(BaseModel):
    city: str
    country: str
    description: str

response = session.answer(
    "What is the capital of France?",
    json_formatting=True,
    pydantic_object=LocationInfo
)

# Using custom tools
tools = [
    {
        "name": "weather",
        "description": "Get current weather",
        "function": get_weather
    }
]
response, tool_calls = session.answer(
    "What's the weather in Paris?",
    tool_list=tools
)

# Streaming responses
for chunk in session.answer(
    "Tell me a long story.",
    stream=True
):
    print(chunk, end="", flush=True)
```

### Vision Models 👁️

```python
from cognicore import ImageAnalyzerAgent

# Initialize the agent
analyzer = ImageAnalyzerAgent()

# Analyze an image
description = analyzer.describe(
    image_path="path/to/image.jpg",
    prompt="Describe this image in detail",
    vllm_provider="groq"
)
print(description)
```

## Available Models 📊

> Note: This list is not exhaustive. The library supports any new model ID released by these providers - you just need to get the correct model ID from your provider's documentation.

### Text Models

| Provider    | Model                           | LLM Provider ID | Model ID                              | Price | Rate Limit (per min) | Context Window | Speed      |
|------------|--------------------------------|----------------|---------------------------------------|-------|---------------------|----------------|------------|
| Google     | Gemini Pro Exp                 | google        | gemini-2.0-pro-exp-02-05             | Free  | 60                  | 32,768        | Ultra Fast |
| Google     | Gemini Flash                   | google        | gemini-2.0-flash                      | Free  | 60                  | 32,768        | Ultra Fast |
| Google     | Gemini Flash Thinking          | google        | gemini-2.0-flash-thinking-exp-01-21   | Free  | 60                  | 32,768        | Ultra Fast |
| Google     | Gemini Flash Lite              | google        | gemini-2.0-flash-lite-preview-02-05   | Free  | 60                  | 32,768        | Ultra Fast |
| GitHub     | O3 Mini                        | github        | o3-mini                               | Free  | 50                  | 8,192         | Fast       |
| GitHub     | GPT-4o                         | github        | gpt-4o                                | Free  | 50                  | 8,192         | Fast       |
| GitHub     | GPT-4o Mini                    | github        | gpt-4o-mini                           | Free  | 50                  | 8,192         | Fast       |
| GitHub     | O1 Mini                        | github        | o1-mini                               | Free  | 50                  | 8,192         | Fast       |
| GitHub     | O1 Preview                     | github        | o1-preview                            | Free  | 50                  | 8,192         | Fast       |
| GitHub     | Meta Llama 3.1 405B            | github        | meta-Llama-3.1-405B-Instruct         | Free  | 50                  | 8,192         | Fast       |
| GitHub     | DeepSeek R1                    | github        | DeepSeek-R1                           | Free  | 50                  | 8,192         | Fast       |
| Groq       | DeepSeek R1 Distill Llama 70B | groq          | deepseek-r1-distill-llama-70b        | Free  | 100                | 131,072       | Ultra Fast |
| Groq       | Llama 3.3 70B Versatile       | groq          | llama-3.3-70b-versatile              | Free  | 100                | 131,072       | Ultra Fast |
| Groq       | Llama 3.1 8B Instant          | groq          | llama-3.1-8b-instant                 | Free  | 100                | 131,072       | Ultra Fast |
| Groq       | Llama 3.2 3B Preview          | groq          | llama-3.2-3b-preview                 | Free  | 100                | 131,072       | Ultra Fast |
| SambaNova  | Llama3 405B                   | sambanova     | llama3-405b                          | Free  | 60                  | 8,000         | Fast       |

### Vision Models

| Provider   | Model                    | Vision Provider ID | Model ID              | Price | Rate Limit (per min) | Speed      |
|-----------|--------------------------|-------------------|----------------------|-------|---------------------|------------|
| Google    | Gemini Vision Exp        | gemini           | gemini-exp-1206      | Free  | 60                | Ultra Fast |
| Google    | Gemini Vision Flash      | gemini           | gemini-2.0-flash     | Free  | 60                | Ultra Fast |
| GitHub    | GPT-4o Vision           | github           | gpt-4o               | Free  | 50                | Fast       |
| GitHub    | GPT-4o Mini Vision      | github           | gpt-4o-mini          | Free  | 50                | Fast       |

### Usage Example with Provider ID and Model ID

```python
from cognicore import LanguageModel

# Initialize a session with specific provider and model IDs
session = LanguageModel(
    model_name="llama-3.3-70b-versatile",  # Model ID from the table above
    provider="groq",                        # Provider ID from the table above
    temperature=0.7
)
```

## Requirements

- Python 3.8 or higher
- Required dependencies will be automatically installed

## Key Features ⭐

- Simple and intuitive session-based interface
- Support for both vision and text models
- Simple configuration with .env file
- Automatic context management
- Tool support for compatible models
- JSON output formatting with Pydantic validation
- Response streaming support
- Smart caching system
- CPU and GPU support

## Contributing 🤝

Contributions are welcome! Feel free to:

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License. See the LICENSE file for details.

## Flexible Configuration ⚡

You can initialize both `LanguageModel` and `ImageAnalyzerAgent` in three ways:

1. **Manual arguments** (classic Python style):
   ```python
   from cognicore import LanguageModel
   llm = LanguageModel(
       model_name="llama-3.3-70b-versatile",
       provider="groq",
       temperature=0.7,
       max_tokens=1024,
   )
   ```
2. **With a configuration dictionary** (useful for programmatic config or dynamic settings):
   ```python
   config = {
       'model_name': 'llama-3.3-70b-versatile',
       'llm_provider': 'groq',
       'temperature': 0.7,
       'max_tokens': 1024,
   }
   llm = LanguageModel(config=config)
   ```
3. **With a YAML config file path** (for reproducibility, sharing, and easy experiment management):
   ```python
   llm = LanguageModel(config="exemple_config.yaml")
   # The YAML file should contain keys like model_name, llm_provider, temperature, etc.
   ```

The same logic applies to `ImageAnalyzerAgent`:
```python
analyzer = ImageAnalyzerAgent(config="exemple_config.yaml")
```

**Why is this useful?**
- You can easily switch between experiments by changing a config file, not your code.
- Share your settings with collaborators or for reproducibility.
- Centralize all your model and generation parameters in one place.
- Use the same config for both text and vision models.

## Multi-Image Support for Vision Models 🖼️🖼️

For some providers (notably **Gemini** and **Groq**), you can pass either a single image path or a list of image paths to the `describe` method:

```python
# Single image
result = analyzer.describe("path/to/image1.jpg", prompt="Describe this image", vllm_provider="gemini")

# Multiple images (Gemini or Groq only)
result = analyzer.describe([
    "path/to/image1.jpg",
    "path/to/image2.jpg"
], prompt="Describe both images", vllm_provider="groq")
```

> **Note:** Passing multiple images is only supported for providers that support it (currently Gemini and Groq). For other providers, only a single image path (str) is accepted.

## Text Classification Utility: `TextClassifier`

`TextClassifier` est une classe utilitaire permettant de classifier un texte parmi une liste de classes définies (index, nom, description). Elle hérite de `LanguageModel` et repose donc sur la même interface flexible (arguments manuels, dictionnaire de config, ou chemin de config YAML).

- **Héritage** : `TextClassifier` hérite de `LanguageModel` pour profiter de toute la logique d'appel LLM multi-provider.
- **Utilisation** : Fournissez un dictionnaire de classes (ou configurez-le dans le YAML), et utilisez la méthode `.classify()` pour obtenir l'index ou le nom de la classe prédite.
- **Prompts** : Les prompts utilisés pour la classification sont stockés dans le dossier `prompts`.
- **Paramètres** : Les paramètres spécifiques à la classification sont à placer dans la section `text_classifier_utils.py` de la config (voir exemple ci-dessous).

### Exemple d'utilisation

```python
from cognicore.text_classifier_utils import TextClassifier

# Utilisation avec un fichier de config YAML
classifier = TextClassifier(config="exemple_config.yaml")
texte = "Je cherche un emploi à Paris."
print("Index de classe :", classifier.classify(texte))
print("Nom de classe :", classifier.classify(texte, return_class_name=True))
```

### Exemple de section config (extrait de `exemple_config.yaml`)

```yaml
# Paramètres pour text_classifier_utils.py
classification_labels_dict: {
  0: {"class_name": "question", "description": "A general question about any topic."},
  1: {"class_name": "job_search", "description": "A request related to job searching or job offers."},
  2: {"class_name": "internet_search", "description": "A request to search for information on the internet."}
}
classifier_system_prompt: "You are an agent in charge of classifying user's queries into different categories of tasks."
query_classification_model: "meta-llama/llama-4-scout-17b-16e-instruct"
query_classification_provider: "groq"
```

- Les paramètres passés explicitement à la classe sont prioritaires sur ceux de la config.
- Les prompts sont à placer dans le dossier `prompts`.

## Image Classification Utility: `ImageClassifier`

`ImageClassifier` est une classe utilitaire permettant de classifier une image parmi une liste de classes définies (index, nom, description). Elle hérite de `ImageAnalyzerAgent` (voir vision_utils.py) et repose donc sur la même interface flexible (arguments manuels, dictionnaire de config, ou chemin de config YAML).

- **Héritage** : `ImageClassifier` hérite de `ImageAnalyzerAgent` pour profiter de toute la logique d'appel vision multi-provider.
- **Utilisation** : Fournissez un dictionnaire de classes (ou configurez-le dans le YAML), et utilisez la méthode `.classify()` pour obtenir l'index ou le nom de la classe prédite pour une image.
- **Prompts** : Les prompts utilisés pour la classification sont stockés dans le dossier `prompts`.
- **Paramètres** : Les paramètres spécifiques à la classification d'image sont à placer dans la section `image_classifier_utils.py` de la config (voir exemple ci-dessous).

### Exemple d'utilisation

```python
from cognicore.image_classifier_utils import ImageClassifier

# Utilisation avec un fichier de config YAML
classifier = ImageClassifier(config="exemple_config.yaml")
image_path = "test_data/chat.jpg"
print("Index de classe :", classifier.classify(image_path))
print("Nom de classe :", classifier.classify(image_path, return_class_name=True))
```

### Exemple de section config (extrait de `exemple_config.yaml`)

```yaml
# Paramètres pour image_classifier_utils.py
image_classification_labels_dict: {
  0: {"class_name": "animal", "description": "Image contenant un animal."},
  1: {"class_name": "ville", "description": "Image représentant une ville ou un lieu urbain."},
  2: {"class_name": "autre", "description": "Tout autre type d'image."}
}
image_classifier_system_prompt: "You are an agent in charge of classifying images into different categories."
image_classification_model: "gemini-1.5-flash"
image_classification_provider: "gemini"
```

- Les paramètres passés explicitement à la classe sont prioritaires sur ceux de la config.
- Les prompts sont à placer dans le dossier `prompts`.

## Internet Search Utility: `InternetSearcher`

`InternetSearcher` is a utility class designed to leverage LLMs with web-browsing capabilities to answer queries based on up-to-date information from the internet. It currently uses Groq's `compound-beta` model, which can access the web.

- **Functionality**: Takes a prompt and returns an answer synthesized from internet search results.
- **Configuration**: Like other utilities in this toolbox, it can be configured via direct arguments, a dictionary, or a YAML file.
- **Streaming**: Supports streaming responses for real-time output.

### Usage Example

```python
from cognicore import InternetSearcher

# Initialize with default settings (Groq's compound-beta)
searcher = InternetSearcher()

# Or initialize with a config file
# searcher = InternetSearcher(config="exemple_config.yaml")

prompt = "What are the latest advancements in battery technology for electric vehicles in 2024?"

# Get a direct answer
result = searcher.search(prompt)
print(result)

# Stream the answer
print("\n--- Streaming ---")
for chunk in searcher.search(prompt, stream=True):
    print(chunk, end="", flush=True)
```

### Configuration in `exemple_config.yaml`

To configure the `InternetSearcher`, you can add the following keys to your YAML file:

```yaml
# Parameters for InternetSearcher
internet_search_model: "compound-beta"
internet_search_provider: "groq"
# You can also override temperature, max_tokens, etc.
```