from typing import List
from langchain.document_loaders.text import TextLoader
from langchain.schema import Document
from loaders.watson_loader.constants import FILE_ENCODING_FORMAT, GIT_DIRECTORY, VALID_CODE_FILE_EXTENSIONS
from loaders.zip_loader.zip_loader import ZipLoader
from utils.file_extension_utils import extract_file_extension


class WatsonLoader(ZipLoader):
    @staticmethod
    def _read_documents(unzipping_directory_path: str) -> List[Document]:
        documents = []
        for directory_path, directory_names, filenames in os.walk(unzipping_directory_path):
            if GIT_DIRECTORY in directory_path:
                continue
            for filename in filenames:
                file_path = os.path.join(directory_path, filename)
                file_extension = extract_file_extension(file_path)
                if not _is_valid_code_file(file_extension):
                    continue
                text_loader = TextLoader(file_path, encoding=FILE_ENCODING_FORMAT)
                documents.extend(text_loader.load())
        return documents


def _is_valid_code_file(file_extension: str) -> bool:
    return file_extension in VALID_CODE_FILE_EXTENSIONS
