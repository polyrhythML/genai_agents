import streamlit as st
import requests

st.title('BHAI - Beyond Human AI')

# Get the state
if 'state' not in st.session_state:
    st.session_state['state'] = []

# Create a form with a text input for the user's message and a button to submit the message
with st.form(key='message_form'):
    user_message = st.text_input('Ask BHAI:')
    submit_button = st.form_submit_button('Send')

if submit_button and user_message.strip() != '':
    # Add the user's message to the conversation history
    st.session_state['state'].append(('User', user_message))

    # Send the user's message to the chatbot
    response = requests.post('http://localhost:8000/predict', json={'input': user_message})
    if response.status_code == 200:
        # Add the chatbot's response to the conversation history
        st.session_state['state'].append(('BHAI', response.json()['response']))
    else:
        st.write('Error: ' + str(response.status_code))

    # If there are more than 10 pairs of messages, remove the first pair
    while len(st.session_state['state']) > 20:
        st.session_state['state'].pop(0)

# Display the conversation history
for role, message in st.session_state['state']:
    if role == 'BHAI':
        st.markdown(f'<p style="color:orange;">{role}: {message}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:white;">{role}: {message}</p>', unsafe_allow_html=True)
