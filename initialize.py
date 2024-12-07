import os
import openai
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from globals import Global
from dbpedia_store import DBpediaStore

def initialize_llm() -> ChatOpenAI:
    """
    Inițializează modelul de limbaj (LLM) folosind API-ul OpenAI.
    
    Returns:
        ChatOpenAI: Instanța modelului de limbaj inițializat.
    """
    os.environ["OPENAI_API_KEY"] = Global.config["OPENAI_API_KEY"]
    openai.organization = Global.config["ORGANIZATION_ID"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Creează o instanță a modelului de limbaj ChatOpenAI
    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )
    return llm

def initialize_vectorstore() -> DBpediaStore:
    """
    Inițializează vectorstore-ul folosind DBpedia și HuggingFaceEmbeddings.
    
    Returns:
        DBpediaStore: Instanța vectorstore-ului inițializat.
    """
    # Setează parametrii pentru modelul de embeddings
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    # Creează o instanță a DBpediaStore folosind embeddings de la HuggingFace
    # Modelul de Embeddings folosit este "sentence-transformers/all-MiniLM-L6-v2",
    # deoarece este un model foarte rapid și eficient

    vectorstore = DBpediaStore(
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        ),
        dbpedia_url=Global.config['DBPEDIA_URL']
    )
    return vectorstore