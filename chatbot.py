import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
from fuzzywuzzy import fuzz, process
import json

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the API key
API_KEY = '12345678'

# Load and preprocess the dataset
try:
    df = pd.read_csv('chat_data.csv', quotechar='"', on_bad_lines='skip')
    responses_dict = dict(zip(df['input'].str.lower(), df['response']))
except pd.errors.ParserError as e:
    logger.error('Error loading CSV file: %s', str(e))
    raise

# Preprocess the dataset for better matching
preprocessed_responses = {}
for key, value in responses_dict.items():
    # Tokenize and normalize responses
    tokens = re.split(r'\W+', key.lower())
    preprocessed_responses[key] = {
        'response': value,
        'tokens': tokens
    }

def preprocess_message(message):
    """
    Preprocess the message for better understanding and normalization.
    """
    message = message.lower().strip()
    return message

def fuzzy_match(message):
    """
    Use fuzzy matching to find the closest match in the responses dictionary.
    """
    message_tokens = re.split(r'\W+', message)
    best_match, score = process.extractOne(message, responses_dict.keys(), scorer=fuzz.partial_ratio)
    if score > 70:  # Adjust the threshold based on requirements
        return responses_dict[best_match]
    return "Sorry, I don't have a response for that."

def respond(message):
    """
    Main response handler using custom dataset for responses.
    """
    message = preprocess_message(message)
    response = fuzzy_match(message)
    return response

@app.route('/chatai', methods=['POST'])
def chat():
    api_key = request.headers.get('Authorization')
    if api_key != f'Bearer {API_KEY}':
        logger.error('Unauthorized access attempt with API key: %s', api_key)
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        message = request.json.get('message', '')
        logger.info('User message: %s', message)
        response = respond(message)
        logger.info('Bot response: %s', response)
        return jsonify({'response': response})
    except Exception as e:
        logger.error('Error processing request: %s', str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=False, threaded=True)
