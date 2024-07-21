from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import logging

def load_qna_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        model = AutoModelForQuestionAnswering.from_pretrained("indolem/indobert-base-uncased")
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return tokenizer, model, pipe
    except (ImportError, OSError) as e:
        logging.error(f"Error loading QnA model: {e}")
        return None, None, None

def ask_question(question: str, context: str):
    tokenizer, model, pipe = load_qna_model()
    if not tokenizer or not model or not pipe:
        return {"error": "No QnA model found"}

    result = pipe(question=question, context=context)
    return result
