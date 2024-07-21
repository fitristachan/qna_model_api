from flask import Flask, request, jsonify
from qna_pipeline import ask_question 
import logging

contexts = [
  {'id': 1, 'context': 'Indonesia dikenal sebagai negara gemah ripah loh jinawi. Sebutan ini muncul karena indonesia negara yang kaya.'},
  {'id': 2, 'context': 'Context 2'},
]

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG) 

@app.route('/')
def index():
    """Liveness check for the service"""
    return jsonify({"status": "SUCCESS"}), 204

@app.route('/qna/<int:context_id>', methods=['GET'])
def get_answer(context_id):
    question = request.args.get('question', type=str)

    if question is None:
        logging.error("Missing question in request")
        return jsonify({'error': 'Missing question in request'}), 400

    logging.debug(f"Received context ID: {context_id}")
    context = next((ctx for ctx in contexts if ctx['id'] == context_id), None)

    if not context:
        logging.error(f"Invalid context ID: {context_id}")
        return jsonify({'error': 'Invalid context ID'}), 400

    return jsonify(ask_question(question=question, context=context['context']))

if __name__ == "__main__":
    app.run(debug=True)


