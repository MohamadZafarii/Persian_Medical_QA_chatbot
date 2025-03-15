# Advanced Medical Chatbot with Groq API Integration

This project represents an **advanced medical question-answering chatbot** that leverages the power of machine learning and natural language processing (NLP) to assist users with medical inquiries. Built with a **custom medical dataset**, the bot utilizes **state-of-the-art** embedding models, **FAISS** vector indexing, **LangChain** for chain-based LLM integration, and the **Groq API** for high-performance computations. This robust system is capable of understanding and responding to user queries based on historical conversations and its embedded medical knowledge.

## Key Features

- **Preprocessing and Vectorization**: The dataset is preprocessed and vectorized to optimize search and retrieval using **FAISS**, enabling high-speed, scalable, and accurate query responses.
- **User Chat History**: The chatbot is designed to keep track of the user's chat history, providing contextually aware responses by leveraging previous conversations.
- **Real-Time Performance**: By integrating the **Groq API**, this system is capable of handling large-scale data processing with incredible speed, making real-time interaction smooth and effective.
- **State-of-the-art NLP**: Built on **Hugging Face**'s cutting-edge models, ensuring top-tier natural language understanding and medical domain relevance.
- **Streamlit Interface**: A user-friendly web interface for seamless interaction with the chatbot.
- **LangChain for LLM Orchestration**: Utilizes **LangChain** for orchestrating interactions with large language models (LLMs), enabling dynamic, multi-step reasoning and response generation.

## Project Structure

- **Dataset Files**:
   - `final_dataset.csv`: The raw medical question-answer dataset.
   - `processed_dataset.csv`: The preprocessed version of the dataset, optimized for vectorization.

- **MAIN Code Files**:
   - `preprocess_dataset.py`: Script for preprocessing and cleaning the raw dataset, making it ready for vectorization.
   - `app.py`: The entry point to the chatbot, serving the model via a web-based **Streamlit** interface.

- **Environment Variables**:
   - The Groq API key is required and must be placed in the `.env` file to authenticate API access.

- **Vector Store**:
   - **FAISS** (Facebook AI Similarity Search) is used for efficient and fast vector storage and retrieval.
   - Embeddings are generated using a pre-trained model from **Hugging Face**, specifically designed for NLP tasks.

- **LangChain Integration**:
   - **LangChain** is used to manage the interaction between large language models (LLMs) and the vector store, enabling sophisticated multi-step reasoning and dynamic context switching.

## Setup Instructions

### 1. Install Dependencies

Ensure that the project dependencies are installed using the following:

```bash
pip install -r requirements.txt
```

2. Preprocess the Dataset
To preprocess the raw dataset, execute the following Python script:

```
python preprocess_dataset.py
```
This step will clean and transform the data from final_dataset.csv into a more usable format, processed_dataset.csv, for further processing and vectorization.

3. Configure Environment Variables
You will need to create a .env file in the project root and add your Groq API Key to the file:

```
GROQ_API_KEY=your_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```
Be sure to replace your_api_key_here with the actual key obtained from Groq.

## 4. Running the Chatbot
To interact with the chatbot, simply run the following command:
```
streamlit run app.py
```
This will launch the chatbot interface in your web browser, where you can start asking medical questions and receive answers in real-time.

## 5. How It Works
**Vectorization & Embeddings:** The medical dataset is vectorized using FAISS to store and efficiently retrieve question-answer pairs. Each medical query is transformed into a high-dimensional vector, which allows the chatbot to find the most relevant response by measuring the distance between vectors.

**Stateful Chatbot:** The chatbot remembers the context of previous conversations. It retrieves relevant information from both the chat history and the vector database to provide the most accurate and contextually appropriate responses.

**NLP and Machine Learning** The system uses a model from Hugging Face trained on medical data, ensuring that the responses are medically accurate and context-aware.

**Groq API Integration:** The use of the Groq API enhances the chatbot's computational efficiency. The API speeds up large data processing, making the system suitable for high-traffic applications or large datasets.

**LangChain:** The integration of LangChain facilitates the orchestration of large language model (LLM) interactions. LangChain enables multi-step reasoning, chain of thought processes, and effective management of LLM prompts to handle more complex medical inquiries. This is critical for enabling the chatbot to break down intricate queries and generate detailed, accurate responses.
## 6. Technologies Used
**Python:** The programming language used for building the chatbot.
**Streamlit:** The web framework used to build the user interface.
**FAISS:** A library developed by Facebook for efficient similarity search and clustering of high-dimensional vectors.
**Hugging Face:** Provides pre-trained NLP models for embedding medical data.
**Groq API:** For accelerating the data processing and model inference in real-time.
**LangChain:** A framework for orchestrating large language model (LLM) interactions, allowing for dynamic reasoning and multi-step prompts in chatbot conversations.

## Model Used
The chatbot leverages the **deepseek-r1-distill-qwen-32b** model from Groq for high-performance reasoning and natural language generation. This model is designed to handle complex, multi-step queries with efficient computations, ensuring accurate and relevant responses in the medical domain.

## Benefits of the System
**Scalable**: The use of FAISS and Groq allows this chatbot to scale effortlessly, processing thousands of queries in real-time without compromising performance.
**Context-Aware**: The chatbot’s memory of user interactions enables it to offer highly relevant and tailored responses, mimicking the flow of a real conversation.
Speed & Efficiency: The integration of Groq's high-performance computing system ensures that the chatbot can process large amounts of data and respond instantaneously, even with complex queries.
**Multi-step Reasoning:** By incorporating LangChain, the chatbot can manage more intricate query flows, break down complex medical problems, and provide well-thought-out answers.
Medical Knowledge Integration: By using pre-trained NLP models fine-tuned on medical datasets, the chatbot provides highly relevant and accurate medical answers.
Technologies Used
**Python**: The programming language used for building the chatbot.
**Streamlit:** The web framework used to build the user interface.
**FAISS:** A library developed by Facebook for efficient similarity search and clustering of high-dimensional vectors.
**Hugging Face:** Provides pre-trained NLP models for embedding medical data.
**Groq API:** For accelerating the data processing and model inference in real-time.
**LangChain:** A framework for orchestrating large language model (LLM) interactions, allowing for dynamic reasoning and multi-step prompts in chatbot conversations.