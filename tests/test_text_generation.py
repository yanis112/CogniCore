from pydantic import BaseModel
from src.language_model import LanguageModel

class TestResponse(BaseModel):
    answer: str

def test_llm_answer_groq():
    """Test Groq model integration"""
    session = LanguageModel(
        model_name="llama-3.3-70b-versatile",
        provider="groq",
        temperature=1.0
    )
    result = session.answer("What is 2+2?")
    assert isinstance(result, str)

def test_llm_answer_gemini():
    """Test Gemini model integration"""
    session = LanguageModel(
        model_name="gemini-2.0-pro-exp-02-05",
        provider="google",
        temperature=1.0
    )
    result = session.answer("What is 2+2?")
    assert isinstance(result, str)

def test_llm_answer_groq_json():
    """Test Groq model with JSON formatting"""
    session = LanguageModel(
        model_name="llama-3.3-70b-versatile",
        provider="groq",
        temperature=1.0
    )
    result = session.answer(
        "Return a JSON with key 'answer' and value '4'",
        json_formatting=True,
        pydantic_object=TestResponse
    )
    assert isinstance(result, dict)
    assert "answer" in result

def test_llm_answer_gemini_json():
    """Test Gemini model with JSON formatting"""
    session = LanguageModel(
        model_name="gemini-2.0-pro-exp-02-05",
        provider="google",
        temperature=1.0
    )
    result = session.answer(
        "Return a JSON with key 'answer' and value '4'",
        json_formatting=True,
        pydantic_object=TestResponse
    )
    assert isinstance(result, dict)
    assert "answer" in result