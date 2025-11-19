# Necessary Imports
import csv
import pandas as pd
import math
import numpy as np
import os
from langsmith import Client, uuid7
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import before_model

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


import os

## Set the OpenAI API key and model name
Open_API_Key = os.environ["OPENAI_API_KEY"]
MODEL="gpt-4o-mini"
summary_llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True)

## Set the Tavily API key, done in .envrc

## Load the vectorstore
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)


## Create the conversational agent

# Creating a retriever
# See https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})



# Define a tool for Amazon product search 
@tool("amazon_product_search")
def amazon_product_search(query: str) -> str:
    """Tool to search for Amazon products."""
    results = retriever.get_relevant_documents(query)
    if not results:
        return "No relevant products found."
    # Format the results for better readability
    formatted_results = "\n".join([f"- {doc.page_content}" for doc in results])
    return f"Here are some relevant Amazon products:\n{formatted_results}"

search_tavily =  TavilySearch(max_results=3,include_answer=True,search_depth="basic").as_tool()


# hwchase17/react is a prompt template designed for ReAct-style
# conversational agents.
client = Client()
prompt = client.pull_prompt("hwchase17/react", include_model=True) # pull "hwchase17/react" prompt from langchain hub


## Create a list of tools: retriever_tool and search_tool
tools = [amazon_product_search, search_tavily] # TODO: Create a list of tools based on search_tavily and amazon_product_search.

# Initialize OpenAI model with streaming enabled
# Streaming allows tokens to be processed in real-time, reducing response latency.
#summary_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, streaming=True)

# Create a ReAct agent
# The agent will reason and take actions based on retrieved tools and memory.


summary_react_agent = create_agent(
    model=summary_llm,
    tools=tools,  # Pass your list of tools here
    system_prompt=prompt.template
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    summary_react_agent,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)


# Building an UI for the chatbot with agents
import gradio as gr

# Define function for Gradio interface
def chat_with_agent(user_input, session_id):
    """Processes user input and maintains session-based chat history."""
    response = agent_with_history.invoke(
        {"messages": [{"role":"user", "content": user_input}]},
        config={"configurable": {"session_id": session_id}}
    )

    # Extract only the 'output' field from the response
    if isinstance(response, dict) and "output" in response:
        return response["output"]  # Return clean text response
    else:
        return response

# Create Gradio app interface
with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")
    session_state = gr.State(value=str(uuid7()))

    submit_button.click(chat_with_agent, inputs=[input_box, session_state], outputs=output_box)

# Launch the Gradio app
app.launch(debug=True, share=True)
