##NLP Utils

# import spacy
# nlp = spacy.load('en_core_web_lg')

def patterns(sents, topic):
    patts = [x for x in sents if topic in x]
    return patts
    # def pattern_check(pattern, doc, name):
    # from spacy.matcher import Matcher
    # matcher = Matcher(nlp.vocab)
    # matcher.add(name, None, pattern)
    # matches = matcher(doc)

    # for match_id, start, end in matches:
    #     print("Match found:", doc[start:end].text)
    # pattern = [{'POS': 'ADJ'}, {'IS_ALPHA': True, 'OP': '?'}, {'IS_ALPHA': True, 'OP': '?'}, {'TEXT': 'NBN'}]


import paralleldots
import sys
sys.path.append("../API")
from creds import PDKEY
paralleldots.set_api_key(PDKEY)

def sentiment(patts):
    snt = paralleldots.batch_sentiment(patts).get('sentiment')
    return snt
    