from abc import ABC, abstractmethod

from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from src.loaders.exception import LoaderError
from src.utils.file_utils import remove_directory, create_dir_with_filename
from src.utils.zip_utils import unzip_files
from src.utils.error_constants import NOT_IMPLEMENTED_MESSAGE

READ_DOCUMENTS_FUNCTION_NAME = "_read_documents"
READ_DOCUMENTS_NOT_IMPLEMENTED_MESSAGE = NOT_IMPLEMENTED_MESSAGE.format(function_name=READ_DOCUMENTS_FUNCTION_NAME)


class ZipLoader(BaseLoader, ABC):
    def __init__(self, zipfile_path: str):
        self._zipfile_path = zipfile_path

    def load(self) -> List[Document]:
        try:
            unzipping_directory_path = create_dir_with_filename(self._zipfile_path)
            unzip_files(self._zipfile_path, unzipping_directory_path)
            documents = self._read_documents(unzipping_directory_path)
            remove_directory(unzipping_directory_path)
        except Exception as error:
            raise LoaderError(str(error))
        return documents

    @staticmethod
    @abstractmethod
    def _read_documents(unzipping_directory_path: str) -> List[Document]:
        raise NotImplementedError(READ_DOCUMENTS_NOT_IMPLEMENTED_MESSAGE)
