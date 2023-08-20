from typing import List, Dict

from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.base import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from src.loaders.watson_loader.watson_loader import WatsonLoader
from src.splitters.watson_splitter import WatsonSplitter

EMBEDDINGS_MODEL_PATH = "krlvi/sentence-t5-base-nlpl-code_search_net"

def build_chain(repo_zipfile_path: str) -> Chain:
    loader = WatsonLoader(repo_zipfile_path)
    documents = loader.load()

    splitter = WatsonSplitter()
    splitted_documents = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(EMBEDDINGS_MODEL_PATH)
    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    llm = OpenAI()

    chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    return chain


