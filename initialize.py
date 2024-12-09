import os
import openai
from langchain_openai import ChatOpenAI
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

def initialize_DBpediaStore() -> DBpediaStore:
    """
    Inițializează clasa DBpediaStore.
    
    Returns:
        DBpediaStore: Instanța clasei DBpediaStore inițializat.
    """

    # Creează o instanță a DBpediaStore 

    dbpediaStore = DBpediaStore(
        dbpedia_url=Global.config['DBPEDIA_URL']
    )
    return dbpediaStore