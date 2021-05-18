##WORDCLOUD UTILS

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
    return words, [lda, tf2, tf_vectorizer]


def wordcloud(words, image_file = "avataaars.png"):
    '''
    Take words & produce a wordcloud plot, in the shape & colouring of an image.
    '''
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    from PIL import Image
    import numpy as np
    
    import spy
    import pathlib
    image_file = pathlib.Path(spy.__file__).parent / 'avataaars.png'
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
    plt.figure(figsize=(12,8), facecolor = (0,0,0)) 
    plt.imshow(firstcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis('off')
    return plt