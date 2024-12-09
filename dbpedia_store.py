from typing import List
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_community.embeddings import HuggingFaceEmbeddings
import spacy

# URL-ul endpoint-ului SPARQL al DBpedia
DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"

class DBpediaStore:
    def __init__(self, dbpedia_url: str = DBPEDIA_SPARQL_URL):
        """
        Inițializează instanța DBpediaStore cu embeddings și URL-ul DBpedia.
        
        Args:
            dbpedia_url (str): URL-ul endpoint-ului SPARQL al DBpedia.
        """
        self.dbpedia_url = dbpedia_url
        self.nlp = spacy.load("en_core_web_sm")  

    def extract_important_terms(self, query: str) -> str:
        """
        Extrage termeni importanți din interogare folosind analiza NLP.
        
        Args:
            query (str): Interogarea utilizatorului.
        
        Returns:
            str: Entitatea sau termenii cei mai relevanți din interogare.
        """
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]  
        if entities:
            return entities[0]  
        
        # Extrage substantivele și substantivele proprii din interogare
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        return " ".join(keywords) if keywords else query  

    def get_dbpedia_context(self, query: str) -> str:
        """
        Recuperează contextul din DBpedia folosind o interogare SPARQL bazată pe inputul utilizatorului.
        
        Args:
            query (str): Interogarea utilizatorului.
        
        Returns:
            str: Contextul recuperat din DBpedia sau un mesaj de eroare.
        """
        processed_query = self.extract_important_terms(query)
        #print(f"Processed Query: {processed_query}")  # Debug: Show processed query
        sparql = SPARQLWrapper(self.dbpedia_url)
        sparql.setQuery(f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?abstract WHERE {{
            ?entity rdfs:label "{processed_query}"@en .
            ?entity dbo:abstract ?abstract .
            FILTER(lang(?abstract) = 'en')
        }} LIMIT 1
        """)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            abstracts = [result["abstract"]["value"] for result in results["results"]["bindings"]]
            return abstracts[0] if abstracts else "Nu există context suplimentar disponibil din DBpedia."
        except Exception as e:
            print(f"Eroare la recuperarea contextului din DBpedia: {e}")
            return "Nu există context suplimentar disponibil din DBpedia."

    def retrieve_context(self, query: str) -> List[str]:
        """
        Recuperează contextul din DBpedia folosind o interogare SPARQL.
        
        Args:
            query (str): Interogarea utilizatorului.
        
        Returns:
            List[str]: Lista abstractelor recuperate din DBpedia.
        """
        # Recuperează contextul din DBpedia
        context = self.get_dbpedia_context(query)
        #print(f"Context recuperat: {context}")

        # Returnează contextul ca o listă cu un singur element
        return [context] if context != "Nu există context suplimentar disponibil din DBpedia." else []
