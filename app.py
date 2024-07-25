from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import re
# import logging

# LOAD MODEL USING TRANSFORMERS LIBRARY
def load_qna_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        model = AutoModelForQuestionAnswering.from_pretrained("arrei/question-answering-uas")
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return tokenizer, model, pipe
    except (ImportError, OSError) as e:
        # logging.error(f"Error loading QnA model: {e}")
        return None, None, None

tokenizer, model, pipe = load_qna_model()
def ask_question(question: str, context: str):
    if not tokenizer or not model or not pipe:
        return {"error": "No QnA model found"}

    result = pipe(question=question, context=context)
    return result



# FLASK API
app = Flask(__name__)

# logging.basicConfig(level=logging.DEBUG) 

@app.route('/')
def index():
    """Liveness check for the service"""
    return jsonify({"status": "Kelompok 2"}), 204

@app.route('/qna', methods=['GET'])
def get_answer():
    question = request.args.get('question', type=str)
    context = request.args.get('context', type=str)

    if question is None:
        # logging.error("Missing question in request")
        return jsonify({'error': 'Missing question in request'}), 400

    if context is None:
        # logging.error("Missing context in request")
        return jsonify({'error': 'Missing context in request'}), 400


    results = ask_question(question=question, context=context)

    r = results['answer']
    r = r.strip()
    r = re.sub(r'[^\w\s]', '', r)
    results['answer'] = r

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)


