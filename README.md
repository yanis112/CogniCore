# Free LLM Toolbox 🚀

A Python package that provides easy-to-use utilities for working with various Language Models (LLMs) and Vision Models. 🎯 But everything is free ! (working on generous free plans of some AI platforms)

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
uv pip install free-llm-toolbox
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
from free_llm_toolbox import LLM_answer_v3

response = LLM_answer_v3(
    prompt="What is the capital of France?",
    model_name="llama2",
    llm_provider="ollama",
    temperature=0.7
)
print(response)
```

### Image Analysis

```python
from free_llm_toolbox import ImageAnalyzerAgent

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
from free_llm_toolbox import LanguageModel

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
from free_llm_toolbox import ImageAnalyzerAgent

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
from free_llm_toolbox import LanguageModel

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