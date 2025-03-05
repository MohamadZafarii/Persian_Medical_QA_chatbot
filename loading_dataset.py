from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
class SplitDataset:
    def __init__(self, dataset_path: str) -> None:
        """
        Initializes the SplitDataset class with the dataset path.

        :param dataset_path: Path to the dataset file (CSV).
        """
        self.dataset_path = dataset_path

    def _load_csv_data(self) -> list:
        """
        Loads the preprocessed data and applies CSV loader.

        :return: A list of documents loaded from the CSV.
        """

        csv_loader = CSVLoader(self.dataset_path , encoding="utf-8")
        return csv_loader.load()

    def split_data(self) -> list:
        """
        Loads the data, splits it into smaller chunks, and returns the split data.

        :return: A list of split documents.
        """
        data = self._load_csv_data()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(data)
