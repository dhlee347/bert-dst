# Python 3+

import re
import itertools

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

def find_sublist(l, sl): # list, sublist
    for i in (i for i, e in enumerate(l) if e == sl[0]):
        if l[i:i+len(sl)] == sl:
            yield i, i+len(sl) # start, exclusive_end

def flatten(x):
    # See https://winterj.me/list_of_lists_to_flatten/
    return list(itertools.chain(*x))

def pad(max_len, value, *lists):
    for x in lists:
        assert len(x) <= max_len
        x.extend([value] * (max_len - len(x)))
    return lists[0] if len(lists) == 1 else lists
