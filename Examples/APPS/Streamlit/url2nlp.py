import streamlit as st
import pandas as pd
import plotly.offline as plotly
import plotly.graph_objs as go

import sys
sys.path.append("../../Utils")
import viz

from streamlit_utils.url_scrape import get_content, get_sentences
from streamlit_utils.wordcloud import ldaify, wordcloud, show_wc

@st.cache()
def get_sents(url):
    sents, title = get_sentences(get_content(url))
    sents = [x.strip() for x in ".".join(sents).split('.')]
    words, model = ldaify(sents)
    return sents, title, words, model


def main():
    st.title('NLP\'ing an article!')
    st.text('''This application extracts text from the article at your url, 
proceeding to produce multiple NLP outputs. Click through and enjoy!''')
    st.sidebar.title('URL Entry')
    url = st.sidebar.text_input('Enter your article\'s url:', value='')
    
    
    if url != '':
        with st.spinner('Extracting Article'):
            sents, title, words, model = get_sents(url)
            st.markdown(f'Article Title: {title}')
            firstcloud, image_colors = wordcloud(words)
            plt = show_wc(firstcloud, image_colors)
            st.sidebar.pyplot(plt)

            st.sidebar.markdown('<hr>', unsafe_allow_html = True)
            st.sidebar.markdown('### Named Entity Analysis')
            ner = st.sidebar.button('NER')
            # st.sidebar.markdown('### Sentiment Analysis')
            # sentmnt = st.sidebar.text_input('Input Topic', value='')
            st.sidebar.markdown('### Topic Analysis')
            topics = st.sidebar.button('LDA Modelling')
            
            if ner:
                from streamlit_utils.ner_extract import ner_e
                combi = " ".join(sents)
                ndf = ner_e(combi)
                
                # from spy import viz
                st.plotly_chart(viz.plotlys(ndf[:10], x = 'Text', y = 'Count', group= 'Likely Entity Type', kind = 'Bar', 
                                            tickangle=90, title = 'NER Counts', xtitle=''))
                st.markdown('<hr>', unsafe_allow_html = True)
                st.write(ndf)
            
            # if sentmnt!='' and not topics and not ner:
            #     # topic = st.text_input('Enter your subject', value='')
            #     # if topic != '':
            #     with st.spinner('Processing Article'):
            #         fig = sent_viz(sents, topic = sentmnt)
            #         st.plotly_chart(fig)
            
            if topics:
                sentmnt=''
                import pyLDAvis
                import pyLDAvis.sklearn
                with st.spinner('Preparing LDA Visual'):
                    ldavis = pyLDAvis.sklearn.prepare(model[0], model[1], 
                                                    model[2], mds='tsne')
                    pyLDAvis.save_html(ldavis, 'hmm2.html')
                    st.markdown('<a href="file:///Users/mitchsa/Documents/ClankR/Examples/NLP/hmm2.html"><ul><li>Right click</li><li>Select "Copy Link Address"</li><li>Open a new tab and paste</li></a>', 
                                unsafe_allow_html = True)

            
            

                

        
def scat_text(pres):
    df = pres.copy()
    group = 'maxcol'
    colours = ['midnightblue','darkmagenta', 'seagreen', 'firebrick','#f4bc61', 
           '#546886', '#4ba4c6','mediumseagreen','lightcoral']   
    coloursIdx = dict(zip(df[group].unique(), colours))
    # pd.options.display.max_colwidth = 500
    # print([text.strip() for text in df.text])
    base = [
                            go.Scatter(
                                opacity = 0.75,
                                x = df[df[group]==val]['negative'],
                                y = df[df[group]==val]['positive'],
                                name = val,
                                marker = dict(size = 16,
                                              color = coloursIdx.get(val),
                                              line=dict(
                                                        color='white',
                                                        width=1,
                                                    )),
                                mode = 'markers',

                                hovertext = [text[:text.rfind(' ', 0, 80)]+"<br>"+text[text.rfind(' ', 0, 80)+1:] if (len(text)>80) & (len(text)<=160)
                                            else text[:text.rfind(' ', 0, 80)]+"<br>"+text[text.rfind(' ', 0, 80)+1:text.rfind(' ', 0, 160)]+"<br>"+text[text.rfind(' ', 0, 160)+1:240] if len(text)>160  
                                            else text 
                                            for text in [text.strip() for text in df[df[group]==val]['text']]],

                                hoverinfo = 'text'
                        ) for val in df[group].unique()

                        ]
    return base

# # @st.cache()
# def sent_viz(sents, topic = 'NBN'):
#     from streamlit_utils.scrape_nlp import patterns, sentiment
#     patts = patterns(sents, topic)
#     show = [x.strip() for x in patts[10:20]]
#     pres = pd.DataFrame(show, columns = ['text'])
#     pres = pd.concat([pres.reset_index()[['text']], pd.DataFrame(sentiment(show))], axis = 1)
#     pres['maxval'] = pres[['negative', 'neutral', 'positive']].max(axis=1)
#     pres['maxcol'] = pres[['negative', 'neutral', 'positive']].idxmax(axis=1)

#     from spy import viz
#     fig = go.Figure(scat_text(pres), viz.plotly_layout(title = 'Sentiment', xtitle = 'Negative', ytitle = 'Positive'))
#     fig.update_layout(hovermode = 'closest')
#     return fig

if __name__ == "__main__":
    main()