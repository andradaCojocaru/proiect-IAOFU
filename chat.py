from typing import List, Tuple
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from globals import Global
from initialize import initialize_llm, initialize_DBpediaStore
# Importă funcția normalize_text din extract.py
from extract import normalize_text


class Chat:
    def __init__(self):
        # Inițializează modelul de limbaj (LLM) și vectorstore-ul
        self.llm = initialize_llm()
        self.DBpediaStore = initialize_DBpediaStore()
        self.isCostar = False
        self.costar = {}

    def updateCostarData(self, type: str, c=None, o=None, s=None, t=None, a=None, r=None):
        """
        Actualizează datele pentru Costar.

        Args:
            type (str): DBPedia sau Costar.
            c (str): Contextul pentru Costar.
            o (str): Obiectivul pentru Costar.
            s (str): Stilul pentru Costar.
            t (str): Tonul pentru Costar.
            a (str): Audiența pentru Costar.
            r (str): Răspunsul pentru Costar.

        Returns:
            None
        """
        if type == "COSTAR":
            self.costar = {
                "CONTEXT": c,
                "OBJECTIVE": o,
                "STYLE": s,
                "TONE": t,
                "AUDIENCE": a,
                "RESPONSE": r
            }
            self.isCostar = True
        else:
            self.isCostar = False

    def create_prompt(self, message: str) -> Tuple[str, List[str]]:
        """
        Creează un prompt pentru modelul de limbaj folosind mesajul utilizatorului și contextul din DBpedia.

        Args:
            message (str): Mesajul utilizatorului.

        Returns:
            Tuple[str, List[str]]: Promptul creat și contextul utilizat.
        """
        context = []
        try:
            # Normalizează mesajul utilizatorului
            normalized_message = normalize_text(message)
            print(f"Performing context retrieval for: {normalized_message}")

            # Recuperează contextul din DBpedia folosind mesajul utilizatorului normalizat
            docs = self.DBpediaStore.retrieve_context(query=normalized_message)
        except Exception as exception:
            print(f"An error occurred during context retrieval: {exception}")
            return str(exception), context

        # Creează promptul adăugând contextul din DBpedia la mesajul utilizatorului
        if self.isCostar:
            prompt = "Objective: " + self.costar['OBJECTIVE'] + '\n' + "Style: " + self.costar['STYLE'] + '\n' + \
                "Tone: " + self.costar['TONE'] + '\n' + "Audience: " + self.costar['AUDIENCE'] + '\n' + \
                "Response: " + self.costar['RESPONSE'] + '\n' + "Context: " + self.costar['CONTEXT'] + '\n' + \
                "Answer the following question: " + normalized_message
        else:
            prompt = Global.config['PROMPT_QUESTION'] + \
                normalized_message + Global.config['PROMPT']
            for doc in docs:
                prompt += doc + '\n'
                context.append(doc)

        print(f"Prompt: {prompt}\n")
        # print(f"Context: {context}\n")

        return prompt, context

    def predict(self, message: str, history: List[List[str]] = None) -> str:
        """
        Generează un răspuns folosind modelul de limbaj și istoricul conversației.

        Args:
            message (str): Mesajul utilizatorului.
            history (List[List[str]], optional): Istoricul conversației. Defaults to None.

        Returns:
            str: Răspunsul generat de modelul de limbaj.
        """
        if history is None:
            history = []
        history_langchain_format = []
        # Adaugă mesajul sistemului la istoricul conversației
        history_langchain_format.append(
            SystemMessage(
                content=Global.config['SYSTEM_PROMPT']))
        for human, ai_response in history:
            # Adaugă mesajele utilizatorului și răspunsurile AI la istoricul conversației
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai_response))
        message, context = self.create_prompt(message)
        history_langchain_format.append(HumanMessage(content=message))

        # Debugging: Print the request being sent to the LLM
        print("Sending request to LLM with the following messages:")
        for msg in history_langchain_format:
            print(f"{msg.__class__.__name__}: {msg.content}")

        try:
            gpt_response = self.llm.invoke(history_langchain_format)
            return gpt_response.content
        except Exception as e:
            print(f"An error occurred while invoking the LLM: {e}")
            return str(e)
