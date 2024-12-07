import gradio as gr
from globals import Global
from chat import Chat

if __name__ == '__main__':
    # Inițializează configurațiile globale din fișierul config.json
    Global.init('config/config.json')

    # Creează o instanță a clasei Chat
    chat = Chat()
    
    # Creează o interfață Gradio pentru chatbot
    # fn=chat.predict: Funcția care va fi apelată pentru a genera răspunsuri
    # inputs="text": Tipul de intrare (text)
    # outputs="text": Tipul de ieșire (text)
    # launch(share=True): Lansează interfața și creează un link public pentru acces
    gr.Interface(fn=chat.predict, inputs="text", outputs="text").launch(share=True)  # Set share=True to create a public link