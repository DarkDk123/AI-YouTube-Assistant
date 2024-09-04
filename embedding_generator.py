# Main Imports
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# For type support.
from typing import List
from langchain_core.documents.base import Document

import os

# Load dotenv
from dotenv import load_dotenv

load_dotenv()

# ________Constants___________
# HF Inference API as LLM
LLM: HuggingFaceEndpoint = HuggingFaceEndpoint(
    repo_id=os.environ["HUGGINGFACE_MODEL_ID"],
    task="text-generation",
    repetition_penalty=1.03,
    temperature=0.8,
)  # type: ignore

HF_EMBEDDER = HuggingFaceEndpointEmbeddings(
    model=os.environ["HF_EMBEDDING_MODEL_ID"],
    task="feature-extraction",
    # huggingfacehub_api_token="my-api-key" (via .env)
)  # type: ignore


def create_vector_db_from_yt_url(video_url: str) -> Chroma:
    """Creates a vector store for transcript text embedding of the given video"""
    # Loading transcript!
    loader: YoutubeLoader = YoutubeLoader.from_youtube_url(video_url)
    transcript: List[Document] = loader.load()

    # Splitting transcript into chunks!
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs: List[Document] = text_splitter.split_documents(transcript)

    # Storing in vector db by converting embeddings!
    # Using Chroma_db and huggingface_embedding model.

    global HF_EMBEDDER

    db_index = Chroma.from_documents(
        docs,
        embedding=HF_EMBEDDER,
    )

    return db_index


def _get_query_similar_docs(db: Chroma, query: str, k=3) -> str:
    """Returns documents most similar to the given query"""

    docs: List[Document] = db.similarity_search(query=query, k=k)  # top-k docs
    entire_doc_context = " | ".join(
        f"Page {i} : {doc.page_content}\n" for i, doc in enumerate(docs)
    )

    return entire_doc_context


def get_response(query: str, db: Chroma):
    docs_context: str = _get_query_similar_docs(db, query=query)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    global LLM
    chain = prompt | LLM
    response = chain.invoke({"question": query, "docs": docs_context})

    return response


# Test embeddings
if __name__ == "__main__":
    db = create_vector_db_from_yt_url(
        "https://youtu.be/_XIihESyy5g?si=mtPgyIV4VDLG6MHh"
    )

    print(_get_query_similar_docs(db, "Something better"))
