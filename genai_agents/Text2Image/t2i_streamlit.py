import streamlit as st
import requests
import json

# The URL of your Flask API endpoint
url = "http://localhost:5000/generate_image"

# Create a text input for the prompt
prompt = st.text_input("Enter your prompt:")

# Create a button to submit the prompt
if st.button("Generate Image"):
    # Prepare the data to send in the POST request
    data = json.dumps({"prompt": prompt})
    
    # Send the POST request
    response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
    
    # Check if the request was successful
    if response.status_code == 200:
        # Display the image
        st.image(response.content, caption="Generated Image", use_column_width=True)
    else:
        st.error(f"Failed to generate image. Status code: {response.status_code}")
