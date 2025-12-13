# standard imports
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# third-party imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver

# load the env variables 
load_dotenv()

# state 
class ChatState(TypedDict):
    """
        ChatState TypedDict to represent the state of the chat.
    Args:
        TypedDict: A dictionary type that maps string keys to values of specified types.
    
    Extra:
    "add_messages": Utility function to add messages to the state which works as "operator.add" but it's optimized for B
    ase-messages or LangGraph messages.
    """
    messages: Annotated[list[BaseMessage], add_messages,"The list of messages exchanged in the chat."]
    

llm_model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# chat_nodes function
def chat_nodes(state: ChatState) -> ChatState:
    """
        chat_nodes function to generate a response from the LLM model based on the current chat state.
    Args:
        state (ChatState): The current state of the chat containing the list of messages.
    Returns:
        ChatState: The updated chat state with the new response message added.
    """
    
    messages = state["messages"]
    
    response = llm_model.invoke(messages)
    
    return {
        "messages": [response]
    }
    
graph = StateGraph(ChatState)

graph.add_node("chat", chat_nodes)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)


checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)


# while True:
#     user_input = input("Type your message: ")

#     if user_input.strip().lower() in ["exit", "bye", "end"]:
#         print("Ending the chat. Goodbye!")
#         break

#     response_state = workflow.invoke({"messages": [HumanMessage(content=user_input)]})

#     message = response_state["messages"][-1].content

#     # Case 1: content is a string
#     if isinstance(message, str):
#         only_text = message

#     # Case 2: content is a list of chunks (LangGraph formatted)
#     elif isinstance(message, list) and len(message) > 0:
#         # find the first text chunk
#         text_chunks = [c["text"] for c in message if c.get("type") == "text"]
#         only_text = text_chunks[0] if text_chunks else ""

#     else:
#         only_text = ""

#     print("AI:", only_text)
