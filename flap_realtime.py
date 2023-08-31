import torch
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import os
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)



# Replace with your Dropbox shared link
dropbox_shared_link = "https://www.dropbox.com/scl/fo/lxwg1ja5e9dkhz8h6fhgc/h?rlkey=naisujy7h4wzi140d8w11cobe&dl=0"

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





base_path = os.path.abspath(os.path.dirname(__file__))  # Assuming this script is in the root of your Streamlit app folder
# Set the path to the model directory within the GitHub repository
model_dir = os.path.join(base_path, "/saved_model_directory")
#path1="Flap.life/saved_model_directory"
logging.debug("Model directory path: %s", model_dir)

dropbox_shared_link = "https://www.dropbox.com/scl/fo/lxwg1ja5e9dkhz8h6fhgc/h?rlkey=naisujy7h4wzi140d8w11cobe&dl=0"

model = BertForSequenceClassification.from_pretrained(dropbox_shared_link)
model.eval()

# Load pre-trained BERT tokenizer
#path2="Flap.life/saved_token_directory"
loaded_tokenizer_path = os.path.join(base_path, "/saved_token_directory") 
tokenizer = BertTokenizer.from_pretrained(loaded_tokenizer_path)

st.title("FLAP Dashboard (DEMO)")
st.write("Enter your essay/personal statement, and we will evaluate it.")

# Input text
input_text = st.text_area("Enter your text here:")

if st.button("Evaluate"):
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
