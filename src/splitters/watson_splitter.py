from typing import List
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from src.utils.file_extension_utils import extract_file_extension

PYTHON_FILE_EXTENSION = '.py'
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 0
FILEPATH_FIELD_NAME_IN_DOCUMENT_METADATA = 'source'


class WatsonSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents: List[Document]) -> List[Document]:
        code_language = _get_code_language(documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        code_splitter = splitter.from_language(code_language)
        splitted_documents = code_splitter.split_documents(documents)
        return splitted_documents


def add_source_path_to_files(splitted_documents: List[Document]) -> None:
    for document in splitted_documents:
        _add_source_path_metadata_to_chunk_as_comment(document)


def _add_source_path_metadata_to_chunk_as_comment(document: Document) -> None:
    document.page_content = f"{SOURCE_FILEPATH_COMMENT} {document.metadata[FILEPATH_FIELD_NAME_IN_DOCUMENT_METADATA]}: \n{document.page_content}"

def _get_code_language(documents: List[Document]) -> Language:
    for document in documents:
        document_file_extension = _extract_document_file_extension(document)
        if PYTHON_FILE_EXTENSION in document_file_extension:
            return Language.PYTHON
    return Language.CPP


def _extract_document_file_extension(document: Document) -> str:
    document_metadata = document.metadata
    document_path = document_metadata[FILEPATH_FIELD_NAME_IN_DOCUMENT_METADATA]
    document_file_extension = extract_file_extension(document_path)

    return document_file_extension
