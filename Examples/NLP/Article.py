import spacy
import string
nlp = spacy.load('en_core_web_lg')
# python -m spacy download en_core_web_lg
# import nltk
# nltk.download('punkt')
from spacy.lang.en.stop_words import STOP_WORDS
sw = list(STOP_WORDS)

from newspaper import Article
import re

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.nlp.stemmers import Stemmer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

def sumy_sum(doc, n=3, Sum = None):
    parser = PlaintextParser.from_string(doc,Tokenizer("english"))
    if Sum is None:
        summariser = LexRankSummarizer() #Stemmer('english') can go inside
    else:
        summariser = Sum
    summary = summariser(parser.document,n)
    result = [str(sentence) for sentence in summary]
    return result

def sent_add(ls):
    df = pd.DataFrame(ls, columns = ['text'])
    
    sid_obj = SentimentIntensityAnalyzer() 
    dicts = df.text.apply(lambda x: sid_obj.polarity_scores(x))
    df = pd.concat([df, dicts.apply(pd.Series)], axis = 1).\
            assign(sentiment=lambda x: np.where((x['compound'] > 0.05), 'Pos', 
                                                np.where(x['compound'] < -0.05, 'Neg', 'Neu')))
    
    return df

def ldaify(dump, num_features = 1000, num_topics = 5):
    '''
    Take a series of sentences, perform text vec & lda, and return a set of words for a wordcloud.
    '''
    # import string
    # hmm2 = dump.translate(str.maketrans('', '', string.punctuation))

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
    tf = tf_vectorizer.fit(dump)
    tf2 = tf.transform(dump)
    tf_feature_names = tf_vectorizer.get_feature_names()

    from sklearn.decomposition import NMF, LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf2)

    words = [tf_feature_names[i] for i in lda.components_[2].argsort()[:-500 - 1 :-1]]
    return words, {'model': lda, 'transform':tf2, 'vector':tf_vectorizer}

def wordclouds(words, image_file = None, image_url = None):
    '''
    Take words & produce a wordcloud plot, in the shape & colouring of an image.
    '''
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    from PIL import Image
    import numpy as np

    import requests
    from io import BytesIO
    if image_file is None:
        if image_url is None:
            im_url = "https://images.pexels.com/photos/1148496/pexels-photo-1148496.jpeg"
        else:
            im_url = image_url
        response = requests.get(im_url, 
                                verify = False)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    elif image_file is not None: 
        img = Image.open(image_file)
    img_coloring = np.array(img)

    firstcloud = WordCloud(mask = img_coloring,
                              stopwords=STOPWORDS,
                              background_color='black',
                              random_state=1
                             ).generate(" ".join(words))
    image_colors = ImageColorGenerator(img_coloring)

    return firstcloud, image_colors
    
def show_wc(firstcloud, image_colors):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(24,8), facecolor = (0,0,0))
    plt.imshow(firstcloud.recolor(color_func=image_colors), interpolation="bilinear")
    # plt.axis('off')
    # return plt

class Art:

    def __init__(self, inp, url = True):
        if url:
            article = Article(inp)
            import warnings
            warnings.simplefilter("ignore")
            article.download()
            article.parse()
            text = article.text
            title = article.title
        else:
            text = inp
        
        if title:
            self.title = title
        else:
            self.title = ''

        self.doc = re.sub('\s+', ' ', 
                        text.replace('\n', ' ').replace("’", "'").\
                            replace('“', '"').replace('”', '"').replace('...', '')).strip()
        self.sents = [x.strip() for x in self.doc.split('. ')]
        
    def ner(self, ent_list = ['GPE', 'ORG', 'PERSON', 'NORP', 'PRODUCT', 'LOC']):
        ##NER
        doc = nlp(self.doc)
        res = list(set([ent for ent in doc.ents if ent.label_ in ent_list]))
        df = pd.DataFrame(columns = ['Text', 'Likely Entity Type'])
        for r in res:
            df = df.append({'Text':r.text, 'Likely Entity Type':r.label_}, ignore_index = True)
        df['Count']=1
        df = df.groupby(['Text', 'Likely Entity Type']).count()[['Count']].reset_index().sort_values('Count', ascending = False)
        self.ner_df = df
    
    def summ(self, n=3, Sum = None):
        ##Summarisation
        summs = sumy_sum(re.sub('\s+', ' ', 
                        self.doc.replace('\n', ' ').replace("’s", "'s")).strip(), 
                        n=n, Sum = Sum)
        summ_df = sent_add(summs)
        syn = []
        for sent in summ_df.text:
            words = nlp(sent)
            deps = [str(words[words[word.i].left_edge.i:words[word.i].right_edge.i+1])
                for word in words if (word.dep_ == 'nsubj') & (word.text not in sw)]
            syn.append(deps)
        summ_df['subjects'] = syn
        self.summ_df = summ_df
    
    def sent(self, topic = ''):
        topic = topic.lower()
        tops = [x for x in self.sents if topic in x.lower()]
        if tops == []:
            # print('Topic not found')
            raise ValueError('Topic not found')
        sent_df = sent_add(tops)
        self.sent_df = sent_df
            
    def wordcloud(self, num_features = 1000, topics = 5, image_file = None, image_url = None):
        words, _ = ldaify([x.strip() for x in self.doc.split()], num_features = num_features, 
                    num_topics = topics)
        firstcloud, image_colours = wordclouds(words)
        self.cloud = firstcloud
        self.c_colours = image_colours

    def lda_mod(self, num_features = 1000, topics = 5):
        _, mod = ldaify([x.strip() for x in self.doc.split()],
                    num_features = num_features, 
                    num_topics = topics)
        self.model = mod
        
    def show_cloud(self):
        show_wc(self.cloud, self.c_colours)


    