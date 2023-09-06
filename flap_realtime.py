import streamlit as st
import os
import logging
import requests
from io import BytesIO
import zipfile
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Replace with your Dropbox shared link
dropbox_shared_link = "https://www.dropbox.com/scl/fo/qr8mndy0nozyvigw9orhj/h?rlkey=bwdsegi6f85cu45uaz5bv5n66&dl=1"
# Temporary directory to store the downloaded and extracted files
temp_dir = "temp_folder"
# Create a directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)
# Function to download and extract the ZIP archive
def download_and_extract_zip(url, output_dir):
    response = requests.get(url)
    with BytesIO(response.content) as zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
# Download and extract the model ZIP file
download_and_extract_zip(dropbox_shared_link, temp_dir)
# Load pre-trained BERT model

#model = BertForSequenceClassification.from_pretrained(temp_dir)
#model.eval()



# Replace with your Dropbox shared link
dropbox_shared_link_token = "https://www.dropbox.com/scl/fo/4tom4ydpwm0ob5isn5ivn/h?rlkey=k61d2k47c3w111vjjef8u7eef&dl=1"
# Temporary directory to store the downloaded and extracted files
temp_dir_token = "temp_folder_token"
# Create a directory if it doesn't exist
os.makedirs(temp_dir_token, exist_ok=True)
# Function to download and extract the ZIP archive
def download_and_extract_zip(url, output_dir):
    response = requests.get(url)
    with BytesIO(response.content) as zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
# Download and extract the model ZIP file
download_and_extract_zip(dropbox_shared_link_token, temp_dir_token)
# Load pre-trained BERT model
#tokenizer = BertTokenizer.from_pretrained(temp_dir_token)



st.title("FLAP Dashboard (DEMO)")
st.write("あなたのエッセイ/個人的な声明を入力してください。私たちがそれを評価します。」(Enter your essay/personal statement, and we will evaluate it for you)")

# Input text
input_text = st.text_area("Enter your text here:")

if st.button("Evaluate"):


    # Save input to a text file with timestamp as the name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"input/{timestamp}.txt"
    with open(input_filename, "w") as file:
        file.write(input_text)


    model = BertForSequenceClassification.from_pretrained(temp_dir)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(temp_dir_token)

    if input_text:
        # Preprocess input
        tokens = tokenizer(input_text, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_probability = probabilities[0][predicted_class].item()

        # Interpret prediction
        if predicted_class == 0:
            prediction_label = "Negative/Rejected"
        else:
            prediction_label = "Positive/Accepted"

        st.write(f"Predicted Label: {prediction_label}")
        st.write(f"Predicted Probability: {predicted_probability:.4f}")
        st.write(f"The model is approximately: {predicted_probability*100 :.4f}% confident that this essay will be {prediction_label}")

    else:
        st.write("Please enter some text.")

# Email input
user_email = st.text_input(
    "あなたのメールアドレスを入力してください。合格するためのエッセイの書き方を教えます。(Enter your email address and we will teach you how to write an essay that will get you accepted )")

if st.button("Submit"):
    # Save email and timestamp to a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("user_emails.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, user_email])
