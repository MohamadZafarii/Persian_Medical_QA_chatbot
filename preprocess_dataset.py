import pandas as pd
import re
from parsivar import Normalizer
import os
from dotenv import load_dotenv
load_dotenv(".env")
dataset_path = os.getenv("DATASET_PATH")
class Preprocess:
    """
    A class to preprocess Persian text data from a CSV file. It includes methods
    for cleaning text such as removing URLs, control characters, and normalizing text.
    """

    def __init__(self, data_path: str, output_path: str) -> None:
        """
        Initializes the Preprocess class with the given dataset path and output path.

        :param data_path: Path to the CSV dataset file.
        :param output_path: Path to save the preprocessed CSV file.
        """
        self.data_path = data_path
        self.output_path = output_path
        self.normalizer = Normalizer()

    def __get_dataset(self) -> pd.DataFrame:
        """
        Reads the CSV dataset from the given path.

        :return: A pandas DataFrame containing the dataset.
        """
        return pd.read_csv(self.data_path, encoding="utf-8")

    def __clean_urls(self, text: str) -> str:
        """
        Removes URLs from the given text.

        :param text: Input text.
        :return: Text with URLs removed.
        """
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def __remove_control_characters(self, text: str) -> str:
        """
        Removes control characters like \n, \t, etc., from the text.

        :param text: Input text.
        :return: Text with control characters removed.
        """
        return re.sub(r'[\t\n\r\f\v]', ' ', text)

    def __normalize_text(self, text: str) -> str:
        """
        Normalizes the text using parsivar's Normalizer.

        :param text: Input text.
        :return: Normalized text.
        """
        return self.normalizer.normalize(text)

    def __remove_extra_spaces(self, text: str) -> str:
        """
        Removes extra spaces and trims the text.

        :param text: Input text.
        :return: Text with extra spaces removed and trimmed.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def clean_text(self, text: str) -> str:
        """
        Cleans the given text by applying URL removal, control character removal,
        extra space trimming, and normalization.

        :param text: Input text.
        :return: Cleaned text.
        """
        text = self.__clean_urls(text)
        text = self.__remove_control_characters(text)
        text = self.__remove_extra_spaces(text)  # Call the new method for removing extra spaces
        text = self.__normalize_text(text)
        return text

    def preprocess_data(self):
        """
        Applies the text cleaning methods to the entire dataset and saves the cleaned data to a CSV file.

        :return: A pandas DataFrame with cleaned text data.
        """
        try:
            data = self.__get_dataset()
            # Apply the cleaning method to each column (series) of the DataFrame
            cleaned_data = data.apply(lambda col: col.apply(self.clean_text))

            cleaned_data.to_csv(self.output_path,index=False)
        except Exception as e:
            raise Exception(f"An error occurred during data preprocessing: {e}")


if __name__=="__main" :
    output_path="./data/processed_dataset.csv"
    prepeocess=Preprocess(data_path=dataset_path,output_path=output_path)
    prepeocess.preprocess_data()
    print("preprocessing Done!")