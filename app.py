import streamlit as st
from dotenv import load_dotenv
import os
from vector_db import VectorDB
from model_init import ModelInit

# Load environment variables
load_dotenv(".env")

# Retrieve necessary environment variables
grok_api = os.getenv("GROQ_API_KEY")
db_path = os.getenv("DB_PATH")
model_name = os.getenv("MODEL_NAME")
embed_name = os.getenv("EMBEDD_NAME")
dataset_path = os.getenv("DATASET_PATH")

# Initialize the vector database
vector_db = VectorDB(embed_name=embed_name, dataset_path=dataset_path, db_path=db_path)

# Initialize the model
model_init = ModelInit(db_path=db_path, groq_api_key=grok_api, model_name=model_name, vector_db=vector_db)
rag_chain = model_init.execute()

# Streamlit UI
st.title("Conversational AI with RAG")

# Session state for chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Text input for user query
user_query = st.text_input("Ask me a question:", "")

if st.button("Submit") and user_query:
    # Get session history
    chat_history = model_init.get_session_history(st.session_state.session_id)

    # Run the RAG chain
    response = rag_chain.invoke({"input": user_query}, config={"configurable": {"session_id": st.session_state.session_id}})

    # Display response
    st.write("**AI Response:**", response["answer"])

    # Store chat history
    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(response["answer"])

# Display chat history
st.subheader("Chat History")
for message in model_init.get_session_history(st.session_state.session_id).messages:
    role = "User" if message.type == "human" else "AI"
    st.write(f"**{role}:** {message.content}")
