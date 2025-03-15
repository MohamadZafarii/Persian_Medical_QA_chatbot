from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from loading_dataset import SplitDataset
import os
import torch

class VectorDB:
    """
    A class to generate and manage a FAISS vector database for document retrieval.
    The database can be persisted to a specified path and loaded later to avoid recomputing embeddings.
    """

    def __init__(self, embed_name: str,dataset_path: str, db_path: str) -> None:
        """
        Initializes the VectorDB class with the given embedding model, device, dataset path, and database path.

        :param embed_name: Name of the embedding model to use (e.g., "distilbert-base-nli-mean-tokens").
        :param device: The device to use for embedding computations (e.g., "cpu" or "cuda").
        :param dataset_path: Path to the dataset CSV file to be used for chunked data.
        :param db_path: Path to save or load the FAISS index for persistence.
        """
        self.embes_name = embed_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = dataset_path
        self.db_path = db_path  # Path to save/load the FAISS database

    def __get_chunked_data(self) -> list:
        """
        Loads and splits the dataset into chunks for embedding processing.

        :return: A list of document chunks (texts) for embedding.
        """
        chunked_data = SplitDataset(dataset_path=self.dataset_path)
        return chunked_data.split_data()


    def get_embedding(self) -> HuggingFaceEmbeddings:
        """
        Initializes the HuggingFaceEmbeddings model with the specified model name and device.

        :return: A HuggingFaceEmbeddings instance for generating document embeddings.
        """
        return HuggingFaceEmbeddings(
            model_name=self.embes_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': False}
        )
    def generate_db(self) -> None:
        """
        Generates or loads the FAISS database from the dataset. If the database already exists at the specified path, it will be loaded;
        otherwise, a new one will be created and saved.

        :return: A FAISS vector store containing the document embeddings.
        """
        # Check if the FAISS index already exists at the specified path
        if os.path.exists(self.db_path):
            print("Loading saved FAISS index...")
            # Load the saved FAISS index
            db = FAISS.load_local(self.db_path, VectorDB.get_embedding())
        else:
            print("Generating new FAISS index...")
            docs = self.__get_chunked_data()
            embedding = VectorDB.get_embedding()
            # Create a new FAISS database
            db = FAISS.from_documents(docs, embedding)
            # Save the FAISS index to the specified path for later use
            db.save_local(self.db_path)



