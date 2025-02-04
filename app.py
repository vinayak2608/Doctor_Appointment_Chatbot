# app.py

from flask import Flask, render_template, request, jsonify
import gtts
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator
from deep_translator import GoogleTranslator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import requests
import random
import os

app = Flask(__name__)

# Conversation history to store messages
conversation_history = []

# Set the folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Tesseract OCR path (update this if Tesseract is installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"'

# In-memory store for past searches (can replace with a DB in production)
past_orders = []
cart = []

# Function to extract text from PDFs
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from images using OCR
def extract_text_from_image(filepath):
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img)
    return text


# Function to send email
def send_email(subject, body, to_email):
    from_email = "roseanddy80@gmail.com"
    from_password = "rjki ncjy ubxu itti"

    # Set up the MIME
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject

    # Attach the body with the msg instance
    message.attach(MIMEText(body, "plain"))

    # Create SMTP session
    session = smtplib.SMTP("smtp.gmail.com", 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(from_email, from_password)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(from_email, to_email, text)
    session.quit()

# Initialize the Hugging Face translation model for Hindi to English
model_name = "Helsinki-NLP/opus-mt-hi-en"


def translate_with_huggingface(input_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Encode and translate the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)

    # Decode and return the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


def translate_with_google(input_text, src_language):
    # Initialize the translator
    translator = Translator()
    translated = translator.translate(input_text, src=src_language, dest='en')
    return translated.text


def translate_with_deep_translator(input_text, src_language):
    try:
        # Perform translation
        translated = GoogleTranslator(source=src_language, target='en').translate(input_text)
        return translated
    except Exception as e:
        # Handle potential errors gracefully
        return f"An error occurred during translation: {e}"


def translate(input_text, language):
    try:
        if language == "hindi":
            return translate_with_deep_translator(input_text, src_language="hi")
        elif language == "tamil":
            return translate_with_deep_translator(input_text, src_language="ta")
        elif language == "telugu":
            return translate_with_deep_translator(input_text, src_language="te")
        else:
            return f"Sorry, the language '{language}' is not supported. Please choose from Hindi, Tamil, or Telugu."
    except Exception as e:
        return f"An error occurred during translation: {e}"

OPENFDA_API_URL = "https://api.fda.gov/drug/label.json"
def search_medicine():
    medicine_name = request.args.get('medicine_name', '').strip()
    if not medicine_name:
        return jsonify({"error": "Medicine name is required"}), 400

    # Query the FDA Drug Label API to get details of the medicine
    search_url = f"{OPENFDA_API_URL}?search=brand_name:\"{medicine_name}\"&limit=1"
    response = requests.get(search_url)

    if response.status_code != 200:
        return jsonify({"error": "Error fetching medicine details"}), 500

    data = response.json()

    if 'results' not in data or len(data['results']) == 0:
        return jsonify({"error": "No results found for the medicine"}), 404

    # Extract relevant drug information from the API response
    drug_info = data['results'][0]

    # Extracting the required details (if available)
    result = {
        "name": drug_info.get("openfda", {}).get("brand_name", ["N/A"])[0],
        "dosage": drug_info.get("dosage_and_administration", "Dosage information not available"),
        "active_ingredient": drug_info.get("active_ingredient", "Active ingredient not available"),
        "description": drug_info.get("description", "No description available")

    }

    # Return the drug details as a JSON response
    return jsonify(result)

def add_to_cart(medicine_name, price):
    cart.append({"medicine": medicine_name, "price": price})

# Function to simulate order submission
def submit_order():
    total_price = sum(item['price'] for item in cart)
    return f"Order placed successfully. Total price: ${total_price}"

@app.route('/')
def home():
    return render_template('index1.html')  # Home page


@app.route('/chat')
def chat():
    return render_template('index.html', conversation=conversation_history)  # Chat page with history


@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['user_message']
    conversation_history.append({'sender': 'user', 'message': user_message})

    # Bot response logic
    if any(keyword in user_message.lower() for keyword in ['hi', 'hello', 'hey']):
        bot_response = "Welcome! How can I help you today? Booking or Rescheduling an appointment?"
    elif any(keyword in user_message.lower() for keyword in
             ['booking', 'scheduling', 'book appointment', 'schedule appointment']):
        bot_response = 'Click here to book your appointment: <a href="/book">Book Appointment</a>'
    elif any(keyword in user_message.lower() for keyword in
             ['reschedule', 'rescheduling', 'change appointment', 'modify appointment']):
        bot_response = 'To reschedule your appointment, please use the same link: <a href="/book">Reschedule Appointment</a>'
    else:
        bot_response = "Sorry, I didn't understand that. Could you rephrase?"

    conversation_history.append({'sender': 'bot', 'message': bot_response})

    # Return the conversation history in the response
    return jsonify({
        'message': bot_response,
        'conversation_history': conversation_history
    })

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text_to_translate = data['text']
    language = data['language']
    translated_text = translate(text_to_translate, language)
    return jsonify({'translated_text': translated_text})

@app.route('/send_email', methods=['POST'])
def send_email_route():
    data = request.json
    translated_message = data['translated_message']
    doctor_email = "sankani14@gmail.com"
    subject = "Patient Symptoms"

    send_email(subject, translated_message, doctor_email)
    return jsonify({'status': 'Email sent successfully'})
@app.route('/book')
def book():
    return render_template('book.html')  # Booking page

@app.route('/pharmacy')
def pharmacy():
    return render_template('pharmacy.html')

@app.route('/billing')
def billing():
    return render_template('billing.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/online')
def online():
    return render_template('online.html')

if __name__ == '__main__':
    app.run(debug=True)
