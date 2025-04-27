from flask import Flask, request, jsonify, render_template
from cipher_utils import detect_cipher, encrypt_message, decrypt_message, detect_language, analyze_frequencies
from cipher_utils import generate_rsa_keys
import numpy as np
from keras.models import load_model
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the lemmatizer for the chatbot
lemmatizer = WordNetLemmatizer()

# Load chatbot data
try:
    intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model10.h5')
except Exception as e:
    print(f"Error loading chatbot resources: {str(e)}")
    # Initialize empty defaults in case files aren't found
    intents = {"intents": []}
    words = []
    classes = []
    model = None


def clean_up_sentence(sentence):
    """Tokenize and lemmatize a sentence for the chatbot"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """Convert a sentence to a bag of words representation"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """Predict the intent class of a sentence"""
    if not model or not words:
        return []

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """Get a response based on the predicted intent"""
    if not intents_list:
        return "I'm not sure what you mean. Can you rephrase that?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm not sure what you mean. Can you rephrase that?"
    return result


# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def index():
    """Main route to display the user interface"""
    return render_template('index.html')


@app.route('/documentation')
def documentation():
    """Route for documentation page"""
    return render_template('documentation.html')


@app.route('/about')
def about():
    """Route for about page"""
    return render_template('about.html')


@app.route('/process', methods=['POST'])
def process():
    """Handle encryption/decryption requests"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing data'}), 400

        text = data['text']
        mode = data.get('mode', 'decrypt')
        decrypt_method = data.get('decrypt_method', 'auto')

        if mode == 'encrypt':
            # Change cipher_type to encrypt_method to match frontend
            cipher_type = data.get('encrypt_method', 'caesar')
            key = data.get('key')

            # Special handling for RSA
            if cipher_type == 'rsa' and not key:
                # Generate a temporary key pair if none is provided
                keys = generate_rsa_keys()
                key = keys['public_key']
                # Store private key for later use
                app.config['TEMP_PRIVATE_KEY'] = keys['private_key']

            result = encrypt_message(text, cipher_type, key)

            response_data = {
                'result': result,
                'cipher_type': cipher_type
            }

            # If RSA was used and keys were generated, include the public key
            if cipher_type == 'rsa' and not data.get('key'):
                response_data['public_key'] = key
                response_data['note'] = 'A temporary RSA key pair was generated. Save the private key for decryption.'
                response_data['private_key'] = app.config['TEMP_PRIVATE_KEY']

            return jsonify(response_data)

        else:  # mode == 'decrypt'
            if decrypt_method == 'auto':
                # Automatic detection
                detected_cipher = detect_cipher(text)
                result = decrypt_message(text, detected_cipher['type'], detected_cipher.get('key'))

                return jsonify({
                    'result': result,
                    'detected_cipher': detected_cipher['type'],
                    'confidence': detected_cipher['confidence'],
                    'language': detect_language(result),
                    'frequencies': analyze_frequencies(result)
                })
            else:
                # Manual decryption method
                key = data.get('key')

                # Handle RSA special case
                if decrypt_method == 'rsa' and not key:
                    if 'TEMP_PRIVATE_KEY' in app.config:
                        key = app.config['TEMP_PRIVATE_KEY']
                    else:
                        return jsonify({'error': 'RSA private key required for decryption'}), 400

                result = decrypt_message(text, decrypt_method, key)

                return jsonify({
                    'result': result,
                    'detected_cipher': decrypt_method,
                    'confidence': 0.7 if decrypt_method != 'auto' else 0.9,
                    'language': detect_language(result),
                    'frequencies': analyze_frequencies(result)
                })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_keys', methods=['POST'])
def generate_keys():
    """Generate RSA key pair for the client"""
    try:
        keys = generate_rsa_keys()
        return jsonify({
            'public_key': keys['public_key'],
            'private_key': keys['private_key']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for chatbot interactions"""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Message missing'}), 400

        user_message = data['message']

        # Get prediction from model
        intents_list = predict_class(user_message)

        # Get response based on prediction
        chatbot_response = get_response(intents_list, intents)

        # Special case for common phrases if the model isn't working well
        if chatbot_response == "I'm not sure what you mean. Can you rephrase that?":
            lower_message = user_message.lower()
            if any(greeting in lower_message for greeting in ['bonjour', 'salut', 'hello', 'hi']):
                chatbot_response = "Bonjour ! Comment puis-je vous aider avec le chiffrement aujourd'hui ?"

            # Add responses for specific cipher types
            elif any(term in lower_message for term in ['caesar', 'césar']):
                chatbot_response = "Le chiffrement de César est une technique simple qui décale chaque lettre d'un certain nombre de positions dans l'alphabet."
            elif any(term in lower_message for term in ['vigenere', 'vigenère']):
                chatbot_response = "Le chiffrement de Vigenère utilise une série de chiffrements de César différents, basés sur les lettres d'un mot-clé."
            elif any(term in lower_message for term in ['rsa', 'asymmetric', 'asymétrique']):
                chatbot_response = "RSA est un algorithme de chiffrement asymétrique qui utilise une paire de clés (publique/privée). Idéal pour l'échange sécurisé de données."
            elif any(term in lower_message for term in ['playfair']):
                chatbot_response = "Le chiffrement Playfair est une technique de chiffrement par substitution manuelle qui utilise une matrice 5x5 basée sur un mot-clé."

        return jsonify({"response": chatbot_response})

    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({"response": "Désolé, je n'ai pas pu traiter votre demande. Veuillez réessayer."}), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """For future visualization features"""
    return jsonify({'message': 'Fonctionnalité de visualisation à venir'})


@app.route('/cipher_info', methods=['GET'])
def cipher_info():
    """Return information about available cipher methods"""
    cipher_info = {
        'caesar': {
            'name': 'Caesar Cipher',
            'description': 'Simple substitution cipher that shifts letters by a fixed number',
            'key_type': 'integer (shift value)',
            'security_level': 'Very Low',
            'example': 'encrypt_message("hello", "caesar", 3) → "khoor"'
        },
        'vigenere': {
            'name': 'Vigenère Cipher',
            'description': 'Polyalphabetic substitution cipher using a keyword',
            'key_type': 'string (keyword)',
            'security_level': 'Low',
            'example': 'encrypt_message("hello", "vigenere", "key") → "rijvs"'
        },
        'base64': {
            'name': 'Base64 Encoding',
            'description': 'Binary-to-text encoding scheme (not encryption)',
            'key_type': 'none',
            'security_level': 'Not secure (encoding only)',
            'example': 'encrypt_message("hello", "base64") → "aGVsbG8="'
        },
        'rsa': {
            'name': 'RSA Encryption',
            'description': 'Asymmetric encryption using public/private key pairs',
            'key_type': 'PEM-formatted public/private keys',
            'security_level': 'Very High',
            'example': 'Keys must be generated first with generate_rsa_keys()'
        },
        'playfair': {
            'name': 'Playfair Cipher',
            'description': 'Manual symmetric encryption using a 5x5 letter matrix',
            'key_type': 'string (keyword)',
            'security_level': 'Medium-Low',
            'example': 'encrypt_message("hello", "playfair", "KEYWORD")'
        }
    }

    return jsonify(cipher_info)


if __name__ == '__main__':
    app.run(debug=True)
