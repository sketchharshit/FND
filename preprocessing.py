import re, nltk, os

_RESOURCES = {"stopwords":"corpora/stopwords","wordnet":"corpora/wordnet","omw-1.4":"corpora/omw-1.4"}
for _n, _p in _RESOURCES.items():
    try: nltk.data.find(_p)
    except LookupError: nltk.download(_n, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

STOPWORDS  = set(stopwords.words("english")) | {"reuters","said"}
LEMMATIZER = WordNetLemmatizer()
STEMMER    = PorterStemmer()

URL_RE         = re.compile(r"https?://\S+|www\.\S+")
HTML_RE        = re.compile(r"<.*?>")
DATELINE_RE    = re.compile(r"^[A-Z][A-Za-z .',/-]{0,60}\(Reuters\)\s*-\s*")
REUTERS_RE     = re.compile(r"\(Reuters\)")
NON_ALPHA_RE   = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

def preprocess_text(text: str, method: str = "lemmatize") -> str:
    if not isinstance(text, str): return ""
    text = DATELINE_RE.sub("", text)
    text = REUTERS_RE.sub("", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if method == "lemmatize":
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif method == "stem":
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

if __name__ == "__main__":
    s = "WASHINGTON (Reuters) - President said on Monday that 2,000 jobs were created! See https://example.com for <b>details</b>."
    print("RAW      :", s)
    print("PROCESSED:", preprocess_text(s))
