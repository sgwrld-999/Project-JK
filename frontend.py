
import streamlit as st

# third-party imports 
from langchain_core.messages import HumanMessage

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
    
    # now we are storing the message in the dictionary before displaying it
    st.session_state.message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
        
    response_state = workflow.invoke({"messages": [HumanMessage(content=user_input)]},config=CONFIG)
    raw_content = response_state["messages"][-1].content
    
    assistant_response = ""
    # handle different content formats from the LLM 
    if isinstance(raw_content, str):
        assistant_response = raw_content
    elif isinstance(raw_content,list) and len(raw_content) > 0:
        # extract the "text" field from the first block 
        assistant_response = raw_content[0].get("text","")
    else:
        assistant_response = ""
    
    # with a call to your LangGraph agent.
    # assistant_response = "Hello! How can I assist you today?"
    st.session_state.message_history.append({'role': 'assistant', 'content': assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
