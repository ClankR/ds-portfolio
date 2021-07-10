import streamlit as st
import pandas as pd
import plotly.express as pe
import os
import sys
import textwrap

sys.path.append("../../NLP")
from Article import Art

@st.cache(allow_output_mutation=True)
def get_sents(url):
    doc = Art(url)
    return doc

def main():
    st.title('NLP\'ing an article!')
    st.text('''This application extracts text from the article at your url, 
proceeding to produce multiple NLP outputs. Click through and enjoy!''')
    st.sidebar.title('URL Entry')
    url = st.sidebar.text_input('Enter your article\'s url:', value='')
    
    if url != '':
        with st.spinner('Extracting Article'):
            doc = get_sents(url)
            st.markdown(f'Article Title: {doc.title}')
            doc.wordcloud()
            st.sidebar.image(doc.cloud.recolor(color_func = doc.c_colours).to_array())
            
            st.sidebar.markdown('<hr>', unsafe_allow_html = True)
            st.sidebar.markdown('### Article Summary')
            summary = st.sidebar.button('Summary')
            st.sidebar.markdown('### Named Entity Analysis')
            ner = st.sidebar.button('NER')
            st.sidebar.markdown('### Sentiment Analysis')
            sentmnt = st.sidebar.text_input('Input Topic', value='')
            st.sidebar.markdown('### LDA Topic Analysis')
            topics = st.sidebar.text_input('Input Number of Topics', value='')
            
            if summary:
                doc.summ()
                st.table(doc.summ_df)

            if ner:
                doc.ner()
                st.plotly_chart(pe.bar(data_frame = doc.ner_df, x = 'Text', 
                                    y = 'Count', color = 'Likely Entity Type', 
                                    title = 'Named Entity Counts'))
                st.markdown('<hr>', unsafe_allow_html = True)
                st.dataframe(doc.ner_df)
            
            if sentmnt!='' and topics=='':
                with st.spinner('Processing Article'):
                    try:

                        doc.sent(sentmnt)
                        temp = doc.sent_df
                        st.write(temp)
                        temp['clean'] = [textwrap.fill(x, 64).replace('\n', '<br>') for x in temp.text]
                        st.plotly_chart(pe.scatter(doc.sent_df, x = 'neg', y = 'pos', 
                                    color = 'sentiment', size = 'neu', hover_name = 'clean'))
                        sentmnt = ''
                    except:
                        st.error(f'Topic {sentmnt} Not Found')
            
            if topics and sentmnt == '' and not ner and not summary:
                import pyLDAvis
                import pyLDAvis.sklearn
                with st.spinner('Preparing LDA Visual'):
                    doc.lda_mod(topics = int(topics))
                    ldavis = pyLDAvis.sklearn.prepare(doc.model.get('model'), 
                                                        doc.model.get('transform'), 
                                                        doc.model.get('vector'), mds='tsne')
                    pyLDAvis.save_html(ldavis, 'LDA.html')
                    st.markdown(f'<a href="file:///{os.getcwd()}/LDA.html"><ol><li>Right click</li><li>Select "Copy Link Address"</li><li>Open a new tab and paste</li></a>', 
                                unsafe_allow_html = True)
                    topics = ''



if __name__ == "__main__":
    main()