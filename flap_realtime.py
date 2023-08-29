import torch
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer

# Load trained model
loaded_model_path = "/saved_model_directory"  # Replace with the actual path
model = BertForSequenceClassification.from_pretrained(loaded_model_path)
model.eval()

# Load pre-trained BERT tokenizer
loaded_tokenizer_path = "/saved_token_directory"  # Replace with the actual path
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
