import streamlit as st
import requests
import time

st.markdown(
    """
    <style>
    .reportview-container {
        font-size:12px;
    }
    h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('BHAI - Beyond Human AI')

# Get the state
if 'state' not in st.session_state:
    st.session_state['state'] = []
if 'message_sent' not in st.session_state:
    st.session_state['message_sent'] = False

# Display the conversation history
for role, message in st.session_state['state']:
    if role == 'BHAI':
        st.markdown(f'<p style="color:orange;">{role}: {message}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:grey;">{role}: {message}</p>', unsafe_allow_html=True)

# Create a form for the user's message
with st.form(key='message_form'):
    user_message = st.text_input('BHAI knows everything, Just Ask:', value='' if st.session_state['message_sent'] else None, key='user_message')
    submit_button = st.form_submit_button('Send')

if (submit_button or user_message != '') and user_message is not None and user_message.strip() != '':
    # Add the user's message to the conversation history
    st.session_state['state'].append(('User', user_message))

    # Send the user's message to the chatbot
    response = requests.put('http://localhost:8000/ask', json={'input': user_message})
    if response.status_code == 200:
        # Simulate typing effect
        agent_response = response.json()['output']

        # Split the response into words
        words = agent_response.split()

        # Group the words into lines of up to 14 words
        lines = [' '.join(words[i:i+14]) for i in range(0, len(words), 14)]

        # Join the lines back together with line breaks
        agent_response_multiline = '\n'.join(lines)

        placeholder = st.empty()
        for i in range(1, len(agent_response_multiline) + 1):
            placeholder.text(agent_response_multiline[:i])
            time.sleep(0.03)

        # Add the chatbot's full response to the conversation history
        st.session_state['state'].append(('BHAI', agent_response_multiline))
    else:
        st.write('Error: ' + str(response.status_code))

    # If there are more than 10 pairs of messages, remove the first pair
    if len(st.session_state['state']) > 20:
        st.session_state['state'].pop(0)
        st.session_state['state'].pop(0)

    # Mark the message as sent
    st.session_state['message_sent'] = True

    # Redraw the conversation history
    st.experimental_rerun()
else:
    # If the message hasn't been sent, clear the flag
    st.session_state['message_sent'] = False
