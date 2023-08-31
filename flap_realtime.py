import streamlit as st
import os
import requests
from io import BytesIO
import zipfile

# Determine the path to the temporary directory within your repository
repo_temp_dir = "Flap.life/"

# Create the repo_temp_dir if it doesn't exist
os.makedirs(repo_temp_dir, exist_ok=True)

# Function to download and extract the ZIP archive
def download_and_extract_zip(url, output_dir):
    response = requests.get(url)
    with BytesIO(response.content) as zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

# Replace with your Dropbox shared link
dropbox_shared_link = "https://www.dropbox.com/scl/fo/lxwg1ja5e9dkhz8h6fhgc/h?rlkey=naisujy7h4wzi140d8w11cobe&dl=1"

st.title("Streamlit App with Downloaded Folder")

if st.button("Download and Extract"):
    st.write("Downloading and extracting the folder...")
    download_and_extract_zip(dropbox_shared_link, repo_temp_dir)
    st.write("Download and extraction complete!")
