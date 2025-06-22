# CogniCore 🚀

<p align="center">
  <img src="assets/CogniCore.png" alt="CogniCore Logo" width="600"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/cognicore/"><img src="https://img.shields.io/pypi/v/cognicore?color=blue&label=PyPI&logo=pypi" alt="PyPI"></a>
  <a href="https://console.groq.com/playground"><img src="https://img.shields.io/badge/Powered%20by-Groq-orange?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHJ4PSI2IiBmaWxsPSIjRkY3RjUwIi8+PHBhdGggZD0iTTIwIDExSDExVjIxSDE3TDIwIDI0VjExWiIgZmlsbD0iI0ZGRiIvPjwvc3ZnPg==" alt="Groq"></a>
  <a href="https://cloud.sambanova.ai/playground"><img src="https://img.shields.io/badge/Powered%20by-SambaNova-yellow?logo=sambanova" alt="SambaNova"></a>
  <a href="https://aistudio.google.com/app/prompts/new_chat"><img src="https://img.shields.io/badge/Powered%20by-Gemini-blue?logo=google" alt="Gemini"></a>
  <a href="https://github.com/marketplace?type=models"><img src="https://img.shields.io/badge/Powered%20by-GitHub-black?logo=github" alt="GitHub"></a>
</p>

A Python package that provides a unified, easy-to-use interface for working with various Language Models (LLMs) and Vision Models from multiple providers. 🎯 It focuses on leveraging the generous free tiers offered by AI platforms.

This project is built on three core principles:

- **🚀 Fast & Cost-Effective Prototyping**: Quickly build and test your ideas by leveraging providers with generous free tiers, minimizing the high costs typically associated with proprietary APIs like OpenAI.
- **🧠 Access to State-of-the-Art Models**: Stay at the cutting edge of AI with curated support for the latest and most powerful open-source and proprietary models as soon as they are released.
- **🧩 Modular & Practical Design**: A clear, feature-rich structure organized into practical modules for vision, text generation, classification, and more, making it easy to integrate advanced AI capabilities into your projects.

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

Before using the library, you need to configure your API keys in a `.env` file. You can get your API keys from the following links:

- [Groq API Key](https://console.groq.com/playground)
- [SambaNova API Key](https://cloud.sambanova.ai/playground)
- [Google Gemini API Key](https://aistudio.google.com/app/prompts/new_chat)
- [GitHub Models Marketplace](https://github.com/marketplace?type=models)

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
    llm_model="gemini-2.0-flash",
    llm_provider="google",
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
    vision_model="llama-3.2-90b-vision-preview",
    vision_provider="groq"
)
print(description)
```

## Usage 🎮

### Text Models 📚

```python
from cognicore import LanguageModel

# Initialize a session with your preferred model
session = LanguageModel(
    llm_model="llama-3-70b",
    llm_provider="groq",
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
    vision_model="llama-3.2-90b-vision-preview",
    vision_provider="groq"
)
print(description)
```

## Available Models 📊

> Note: This list is not exhaustive. The library supports any new model ID released by these providers - you just need to get the correct model ID from your provider's documentation.

### Text Models

| Provider    | Model                           | LLM Provider ID | Model ID                              | Price | Rate Limit (per min) | Context Window | Speed      |
|------------|--------------------------------|----------------|---------------------------------------|-------|---------------------|----------------|------------|
| SambaNova  | DeepSeek R1 670B               | sambanova      | DeepSeek-R1-0528                      | Free  | 60                  | 32,000         | Ultra Fast |
| SambaNova  | Llama3 405B                   | sambanova      | llama3-405b                           | Free  | 60                  | 8,000          | Fast       |
| GitHub     | Meta Llama 3.1 405B            | github         | meta-Llama-3.1-405B-Instruct          | Free  | 50                  | 8,192          | Fast       |
| Google     | Gemini 2.5 Pro                 | google         | gemini-2.5-pro-preview-05-06          | Free  | 60                  | 32,768         | Ultra Fast |
| GitHub     | GPT-4.1                        | github         | openai/gpt-4.1                        | Free  | 50                  | 8,192          | Fast       |
| GitHub     | GPT-4o                         | github         | gpt-4o                                | Free  | 50                  | 8,192          | Fast       |
| GitHub     | O1 Preview                     | github         | o1-preview                            | Free  | 50                  | 8,192          | Fast       |
| Groq       | DeepSeek R1 Distill Llama 70B | groq           | deepseek-r1-distill-llama-70b         | Free  | 100                 | 131,072        | Ultra Fast |
| Groq       | Llama 3.3 70B Versatile       | groq           | llama-3.3-70b-versatile               | Free  | 100                 | 131,072        | Ultra Fast |
| Groq       | Qwen3 32B                      | groq           | qwen/qwen3-32b                        | Free  | 100                 | 4,096          | Ultra Fast |
| Groq       | Llama 4 Maverick 17B           | groq           | llama-4-maverick-17b-128e-instruct    | Free  | 100                 | 131,072        | Ultra Fast |
| GitHub     | DeepSeek R1                    | github         | DeepSeek-R1                           | Free  | 50                  | 8,192          | Fast       |
| Google     | Gemini 2.5 Flash               | google         | gemini-2.5-flash-preview-05-20        | Free  | 60                  | 32,768         | Ultra Fast |
| Google     | Gemma 3N E4B IT                | google         | gemma-3n-e4b-it                       | Free  | 60                  | 32,768         | Ultra Fast |
| Google     | Gemini Pro Exp                 | google         | gemini-2.0-pro-exp-02-05              | Free  | 60                  | 32,768         | Ultra Fast |
| Google     | Gemini Flash                   | google         | gemini-2.0-flash                      | Free  | 60                  | 32,768         | Ultra Fast |
| Google     | Gemini Flash Thinking          | google         | gemini-2.0-flash-thinking-exp-01-21   | Free  | 60                  | 32,768         | Ultra Fast |
| Google     | Gemini Flash Lite              | google         | gemini-2.0-flash-lite-preview-02-05   | Free  | 60                  | 32,768         | Ultra Fast |
| Groq       | Llama 3.1 8B Instant          | groq           | llama-3.1-8b-instant                  | Free  | 100                 | 131,072        | Ultra Fast |
| Groq       | Llama 3.2 3B Preview          | groq           | llama-3.2-3b-preview                  | Free  | 100                 | 131,072        | Ultra Fast |
| GitHub     | GPT-4o Mini                    | github         | gpt-4o-mini                           | Free  | 50                  | 8,192          | Fast       |
| GitHub     | O3 Mini                        | github         | o3-mini                               | Free  | 50                  | 8,192          | Fast       |
| GitHub     | O1 Mini                        | github         | o1-mini                               | Free  | 50                  | 8,192          | Fast       |

### Vision Models

| Provider   | Model                    | Vision Provider ID | Model ID              | Price | Rate Limit (per min) | Speed      |
|-----------|--------------------------|-------------------|----------------------|-------|---------------------|------------|
| Google    | Gemini 2.5 Pro Vision    | gemini           | gemini-2.5-pro-preview-05-06 | Free  | 60                | Ultra Fast |
| GitHub    | GPT-4.1 Vision           | github           | openai/gpt-4.1       | Free  | 50                | Fast       |
| GitHub    | GPT-4o Vision           | github           | gpt-4o               | Free  | 50                | Fast       |
| GitHub    | Phi-4 Multimodal         | github           | phi-4-multimodal-instruct | Free  | 50                | Fast       |
| Groq      | Llama 4 Maverick Vision  | groq             | meta-llama/llama-4-maverick-17b-128e-instruct | Free | 100 | Ultra Fast |
| Groq      | Llama 4 Scout Vision     | groq             | meta-llama/llama-4-scout-17b-16e-instruct | Free | 100 | Ultra Fast |
| Google    | Gemini 2.5 Flash Vision  | gemini           | gemini-2.5-flash-preview-05-20 | Free  | 60                | Ultra Fast |
| Google    | Gemini 3N E4B IT Vision  | gemini           | gemini-3n-e4b-it     | Free  | 60                | Ultra Fast |
| Google    | Gemini Vision Exp        | gemini           | gemini-exp-1206      | Free  | 60                | Ultra Fast |
| Google    | Gemini Vision Flash      | gemini           | gemini-2.0-flash     | Free  | 60                | Ultra Fast |
| GitHub    | GPT-4o Mini Vision      | github           | gpt-4o-mini          | Free  | 50                | Fast       |

### Usage Example with Provider ID and Model ID

```python
from cognicore import LanguageModel

# Initialize a session with specific provider and model IDs
session = LanguageModel(
    llm_model="llama-3.3-70b-versatile",  # Model ID from the table above
    llm_provider="groq",                  # Provider ID from the table above
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
       llm_model="llama-3.3-70b-versatile",
       llm_provider="groq",
       max_tokens=1024,
   )
   ```
2. **With a configuration dictionary** (useful for programmatic config or dynamic settings):
   ```python
   config = {
       'llm_model': 'llama-3.3-70b-versatile',
       'llm_provider': 'groq',
       'max_tokens': 1024,
   }
   llm = LanguageModel(config=config)
   ```
3. **With a YAML config file path** (for reproducibility, sharing, and easy experiment management):
   ```python
   llm = LanguageModel(config="exemple_config.yaml")
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
result = analyzer.describe("path/to/image1.jpg", prompt="Describe this image", vision_model="gemini-2.5-flash-preview-05-20", vision_provider="gemini")

# Multiple images (Gemini or Groq only)
result = analyzer.describe([
    "path/to/image1.jpg",
    "path/to/image2.jpg"
], prompt="Describe both images", vision_model="llama-3.2-90b-vision-preview", vision_provider="groq")
```

> **Note:** Passing multiple images is only supported for providers that support it (currently Gemini and Groq). For other providers, only a single image path (str) is accepted.

## Text Classification Utility: `TextClassifier`

`TextClassifier` is a utility class for classifying text into a defined list of classes (index, name, description). It inherits from `LanguageModel` and thus relies on the same flexible interface (manual arguments, config dictionary, or YAML config path).

- **Inheritance** : `TextClassifier` inherits from `LanguageModel` to leverage all the multi-provider LLM calling logic.
- **Usage** : Provide a class dictionary (or configure it in the YAML), and use the `.classify()` method to get the predicted class index or name.
- **Prompts** : The prompts used for classification are stored in the `prompts` folder.
- **Parameters** : Parameters specific to classification should be placed in the `text_classifier_utils.py` config section (see example below).

### Usage Example

```python
from cognicore.text_classifier_utils import TextClassifier

# Using a YAML config file
classifier = TextClassifier(config="exemple_config.yaml")
text = "I'm looking for a job in Paris."
print("Class index:", classifier.classify(text))
print("Class name:", classifier.classify(text, return_class_name=True))
```

### Example config section (from `exemple_config.yaml`)

```yaml
# Parameters for text_classifier_utils.py
classification_labels_dict: {
  0: {"class_name": "question", "description": "A general question about any topic."},
  2: {"class_name": "internet_search", "description": "A request to search for information on the internet."}
}
classifier_system_prompt: "You are an agent in charge of classifying user's queries into different categories of tasks."
```

- Prompts should be placed in the `prompts` folder.

## Image Classification Utility: `ImageClassifier`

`ImageClassifier` is a utility class for classifying an image among a defined list of classes (index, name, description). It inherits from `ImageAnalyzerAgent` (see vision_utils.py) and thus relies on the same flexible interface (manual arguments, config dictionary, or YAML config path).

- **Inheritance** : `ImageClassifier` inherits from `ImageAnalyzerAgent` to leverage all the multi-provider vision calling logic.
- **Usage** : Provide a class dictionary (or configure it in the YAML), and use the `.classify()` method to get the predicted class index or name for an image.
- **Prompts** : The prompts used for classification are stored in the `prompts` folder.
- **Parameters** : Parameters specific to image classification should be placed in the `image_classifier_utils.py` config section (see example below).

### Usage Example

```python
from cognicore.image_classifier_utils import ImageClassifier

# Using a YAML config file
image_classifier = ImageClassifier(config="exemple_config.yaml")
image_path = "path/to/image.jpg"
print("Class index:", image_classifier.classify(image_path))
print("Class name:", image_classifier.classify(image_path, return_class_name=True))
```

### Example config section (from `exemple_config.yaml`)

```yaml
# Parameters for image_classifier_utils.py
classification_labels_dict: {
  0: {"class_name": "cat", "description": "A domestic cat."},
  1: {"class_name": "dog", "description": "A domestic dog."},
  2: {"class_name": "bird", "description": "A bird."}
}
image_classifier_system_prompt: "You are an agent in charge of classifying images into different categories."
image_classification_model: "meta-llama/llama-4-scout-17b-16e-instruct"
image_classification_provider: "groq"
```

- Parameters passed explicitly to the class take precedence over those in the config.
- Prompts should be placed in the `prompts` folder.