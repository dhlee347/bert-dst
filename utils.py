# Python 3+

import re

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def tokenize(text):
    """ "I'm a boy." --> ['I', "'", 'm', 'a', 'boy', '.'] """
    text_lower = convert_to_unicode(text).lower()
    return [tok for tok in map(str.strip, re.split("(\W+)", text_lower)) \
            if len(tok) > 0]
