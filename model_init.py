from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from vector_db import VectorDB
from typing import Optional


# Store for session histories
store = {}


class ModelInit:
    """
    A class to initialize and manage the model and database for a conversational AI system.
    The class handles the creation of the retriever and the chain for question answering.

    Attributes:
        db_path (str): Path to the local FAISS database.
        groq_api_key (str): API key for the Groq model.
        model_name (str): Name of the Groq model.
        vector_db (VectorDB): Instance of VectorDB to handle embeddings.
        db (Optional[FAISS]): FAISS instance loaded from the local database.
        model (Optional[ChatGroq]): ChatGroq model instance.
        contextualize_q_prompt (ChatPromptTemplate): The prompt used for contextualizing the question.
        system_prompt (str): The system-level instructions for the question-answering task.
        qa_prompt (ChatPromptTemplate): The prompt used for the question-answering task.
    """

    def __init__(self, db_path: str, groq_api_key: str, model_name: str, vector_db: VectorDB) -> None:
        """
        Initializes the ModelInit instance with necessary parameters.

        Args:
            db_path (str): Path to the FAISS database.
            groq_api_key (str): API key for Groq model.
            model_name (str): Name of the Groq model.
            vector_db (VectorDB): Instance of the VectorDB class for embeddings.
        """
        self.db_path = db_path
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.vector_db = vector_db
        self.db: Optional[FAISS] = None
        self.model: Optional[ChatGroq] = None
        self.contextualize_q_prompt = self._create_contextualize_q_prompt()
        self.system_prompt = self._create_system_prompt()
        self.qa_prompt = self._create_qa_prompt()

    def _load_db(self) -> FAISS:
        """
        Loads the FAISS database if not already loaded.

        Returns:
            FAISS: The loaded FAISS database.
        """
        if not self.db:
            self.db = FAISS.load_local(
                folder_path=self.db_path,
                embeddings=self.vector_db.get_embedding(),
                allow_dangerous_deserialization=True
            )
        return self.db

    def _create_model(self) -> ChatGroq:
        """
        Creates and initializes the ChatGroq model if not already created.

        Returns:
            ChatGroq: The created ChatGroq model.
        """
        if not self.model:
            self.model = ChatGroq(model_name=self.model_name, groq_api_key=self.groq_api_key)
        return self.model

    def _create_contextualize_q_prompt(self) -> ChatPromptTemplate:
        """
        Creates the contextualize question prompt template.

        Returns:
            ChatPromptTemplate: The contextualize question prompt template.
        """
        prompt = (
            "Given the user's latest question and the chat history, "
            "rephrase the question if necessary so that it can be understood independently of the previous conversation. "
            "Do not answer the question; simply ensure that it stands on its own and can be understood without context."
        )
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def _create_system_prompt(self) -> str:
        """
        Creates the system prompt for answering questions.

        Returns:
            str: The system prompt.
        """
        return (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. "
            "Use one paragraph  maximum and keep the answer concise."
            "answer the question in persian."
            "the thinking process be in persian"
            "\n\n"
        )

    def _create_qa_prompt(self) -> ChatPromptTemplate:
        """
        Creates the question-answering prompt template.

        Returns:
            ChatPromptTemplate: The question-answering prompt template.
        """
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "Context: {context}")
        ])

    def history_aware_retriever(self) -> RunnableWithMessageHistory:
        """
        Creates and returns a history-aware retriever.

        Returns:
            RunnableWithMessageHistory: A retriever that is aware of chat history.
        """
        self._load_db()
        self._create_model()
        retriever = self.db.as_retriever(search_type="similarity",search_kwargs={"k":6})
        return create_history_aware_retriever(retriever=retriever, prompt=self.contextualize_q_prompt,llm=self.model)

    def generate_chain(self) -> RunnableWithMessageHistory:
        """
        Generates the retrieval-augmented generation (RAG) chain using the question-answering chain.

        Returns:
            RunnableWithMessageHistory: The RAG chain that performs retrieval and answering.
        """
        self._load_db()  # Ensure database is loaded
        self._create_model()  # Ensure model is initialized

        qa_chain = create_stuff_documents_chain(self.model, self.qa_prompt)  # `self.model` won't be None now
        rag_chain = create_retrieval_chain(self.history_aware_retriever(), qa_chain)
        return rag_chain
    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves the session history for a given session ID.

        Args:
            session_id (str): The session ID.

        Returns:
            BaseChatMessageHistory: The chat history associated with the session ID.
        """
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def execute(self) -> RunnableWithMessageHistory:
        """
        Executes the RAG chain with message history.

        Returns:
            RunnableWithMessageHistory: The executable object that runs the RAG chain with message history.
        """
        conversational_rag_chain = RunnableWithMessageHistory(
            self.generate_chain(),
            ModelInit.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain


# Store for session histories
store = {}


