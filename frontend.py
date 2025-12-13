import streamlit as st

# third-party imports 
from langchain_core.messages import HumanMessage, AIMessageChunk

# custom imports 
from backend import workflow

"""
Now, to store the files we'll store in the dictionary as follows:
    {'role' : user, 'content' : 'Hi'}
    {'role' : assistant , 'content' : 'Hello'}
"""

CONFIG = {'configurable' : {'thread_id' : 'thread-1'}}

""" 
Problem:
Same problem as the previous iteration, the messages are not persistent.
Once we enter a new message, the previous messages disappear or message history is lost.

To overcome the problem, we will use session state to store the message history.
"""
st.title("LangGraph Chatbot")


# Initialize message history in session state if it doesn't exist
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# before taking new input, we will display the previous messages
for message in st.session_state.message_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        
user_input = st.chat_input("You: ")

if user_input:
    
    # store the user message
    st.session_state.message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # stream the response chunk by chunk using stream_mode="messages"
        for chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                content = chunk.content
                
                # Check if content is a list (common with Gemini)
                if isinstance(content, list):
                    # Extract text from each block in the list
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            full_response += block["text"]
                        elif isinstance(block, str):
                            full_response += block
                # Check if content is a simple string
                elif isinstance(content, str):
                    full_response += content
                
                # update the placeholder with the text so far + a cursor
                message_placeholder.markdown(full_response + "▌")