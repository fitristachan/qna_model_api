import asyncio
import aiohttp
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from flask import Flask, request, jsonify
import logging

# CONTEXTS
contexts = [
  {'id': 1, 'context': 'Indonesia dikenal sebagai negara gemah ripah loh jinawi. Sebutan ini muncul karena indonesia negara yang kaya.'},
  {'id': 2, 'context': 'Context 2'},
]

# Function to download and load the model from a remote URL asynchronously
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.read()

async def load_model_from_url(model_url, tokenizer_url):
    try:
        async with aiohttp.ClientSession() as session:
            model_task = asyncio.create_task(fetch(session, model_url))
            tokenizer_task = asyncio.create_task(fetch(session, tokenizer_url))

            model_data = await model_task
            tokenizer_data = await tokenizer_task

            # Save model and tokenizer to disk
            with open('model.bin', 'wb') as model_file:
                model_file.write(model_data)
            
            with open('tokenizer.json', 'wb') as tokenizer_file:
                tokenizer_file.write(tokenizer_data)

            tokenizer = AutoTokenizer.from_pretrained('tokenizer.json')
            model = AutoModelForQuestionAnswering.from_pretrained('model.bin')
            pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
            return tokenizer, model, pipe
    except (ImportError, OSError) as e:
        logging.error(f"Error loading QnA model from URL: {e}")
        return None, None, None

# Load model asynchronously at startup
async def initialize_model():
    global tokenizer, model, pipe
    tokenizer, model, pipe = await load_model_from_url('model_url', 'tokenizer_url')

# Flask API
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

    result = pipe(question=question, context=context['context'])
    return jsonify(result)

if __name__ == "__main__":
    # Initialize the model asynchronously
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_model())
    app.run(debug=True)
