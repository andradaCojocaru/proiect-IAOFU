import re
import string
from globals import Global

def normalize_text(text: str) -> str:
    """The function normalizes the text by removing links, hashtags, usernames (in case they appear),
    makes the whole transcript lowercase and remove the punctuation.
    Args:
    text (str): The initial text before normalization
    Returns:
    str: The text after normalization"""
    patterns = {
        r'http\S+': 'LINK',
        r'@\S+': 'USERNAME',
        r'#\S+': 'HASHTAG'
    }
    text = text.translate(str.maketrans('', '', string.punctuation))
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    return text

