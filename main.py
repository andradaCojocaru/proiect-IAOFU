
from globals import Global
from chat import Chat
from gui import handleGUI

if __name__ == '__main__':
    # Inițializează configurațiile globale din fișierul config.json
    Global.init('config/config.json')

    # Creează o instanță a clasei Chat
    chat = Chat()

    # Creeaza o interfata grafica pentru chatbot
    handleGUI(chat)
