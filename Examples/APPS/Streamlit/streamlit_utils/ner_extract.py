import spacy
nlp = spacy.load('en_core_web_lg')
import pandas as pd

def ner_e(combi):
    doc = nlp(combi)
    res = list(set([ent for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON', 'NORP', 'PRODUCT', 'LOC']]))
    df = pd.DataFrame(columns = ['Text', 'Likely Entity Type'])
    for r in res:
        df = df.append({'Text':r.text, 'Likely Entity Type':r.label_}, ignore_index = True)
    df['Count']=1
    df = df.groupby(['Text', 'Likely Entity Type']).count()[['Count']].reset_index().sort_values('Count', ascending = False)
    return df