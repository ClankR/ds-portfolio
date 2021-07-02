import streamlit as st

import pyspark
from pyspark.sql import functions as func, types as typ 
from pyspark.sql.window import Window 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pandas as pd
from utils import viz

def main():
    st.title('Steamlit it up!')
    st.text('''This application leverages a spark connection for processing, 
    plotly for visuals, as well as video embeds.''')
    st.sidebar.title('Options')
    #Spark
    with st.spinner('Spark Setup in Progress...'):
        sdf, url = data()
    
    #Spark UI
    st.markdown('<a href = "{}" target = "_blank">Click here to open Spark UI</a>'.format(url),
            unsafe_allow_html=True)

    ##Main
    num_cols = [t[0] for t in sdf.dtypes[1:] if t[1] == 'int' or t[1] == 'double']
    choice = None
    choice = st.sidebar.selectbox(options = ['Select', 'Univariate Analysis', 'Correlation Analysis', 'What does this button do...'],
                                    label = 'Analysis Type')

    if choice== 'Select':
        st.sidebar.markdown('Please Make a Selection')
        st.text('Please Make a Selection on the left.')
    #Univariate
    elif choice == 'Univariate Analysis':
        feat = st.sidebar.selectbox(options = ['Select...']+num_cols, label = 'Feature Choice')
        if feat == 'Select...':
            st.sidebar.markdown('Please Select a Feature')
        else:
            with st.spinner('Building Graph'):
                fig = univariate(sdf, feat)
            st.markdown('''<hr>Comparing Churn Populations''', unsafe_allow_html=True)
            st.plotly_chart(fig, height = 500, width = 800)    

    elif choice == 'Correlation Analysis':
        with st.spinner('Building Correlation Matrix'):
            fig = correlate(sdf, num_cols)    
        # plotly.iplot(fig)
        st.plotly_chart(fig, height = 800, width = 800)
        # fig
    
    elif choice == 'What does this button do...':
        # st.video(data = 'https://www.youtube.com/watch?v=LlhKZaQk860')
        st.markdown('''<iframe width="600" height="400"
                        src="https://www.youtube.com/embed/1Bix44C1EzY?autoplay=1">
                        </iframe>''', unsafe_allow_html=True)
        
def spark_connect():
    sc_conf = SparkConf()
    ##Many more config options are available - set depending on your job!
    sc_conf.setAll([('spark.executor.cores', 4 )]).setAppName('STTest')

    spark = SparkSession.builder.config(conf = sc_conf).getOrCreate()
    return spark

@st.cache(allow_output_mutation=True)    
def data():
    spark = spark_connect()
    sc = spark.sparkContext
    url = sc._jsc.sc().uiWebUrl().get()
    sdf = spark.read.csv('Data/cell2celltrain.csv', 
                        inferSchema = True, header = True, nullValue = 'NA')

    return sdf, url

def univariate(sdf, feat):
    group = viz.spark_continuous_viz(sdf.filter(func.col(feat)<200), feat, plot = False, result_name = feat,
                                            description = False, transformed_df = True).\
                groupby(['Churn', feat]).count().\
                withColumn('perc', 
                        (func.col('count')/
                            func.sum(func.col('count')).\
                                over(Window().partitionBy('Churn')))*100).\
                toPandas()
            
    fig = viz.plotlys(group, x = feat, y = 'perc', kind = 'Bar', group = 'Churn')
    return fig

def correlate(sdf, num_cols):
    from pyspark.mllib.stat import Statistics
    from utils import viz

    # Need to remove nulls - should impute realistically but for quick analysis...
    df = sdf.select(num_cols[3:]).dropna()

    #Create Correlation Matrix
    col_names = df.columns
    features = df.rdd.map(lambda row: row[0:])
    corr_mat=Statistics.corr(features, method="pearson")
    corr_df = pd.DataFrame(corr_mat)
    corr_df.index, corr_df.columns = col_names, col_names

    #Copy to overcome read-only issues when replacing diagonal duplicates
    corrdf = corr_df.copy()
    for col in corrdf.columns:
        for i in range(0, corrdf.index.get_loc(col)+1):
            corrdf.loc[corrdf.index[i], col]=None

    import plotly.offline as plotly
    import plotly.graph_objs as go
    from PIL import Image
    import pathlib

    data =  [go.Heatmap(z = corrdf,
                    x = corrdf.columns,
                    y = [y+'  ' for y in corrdf.index],
                    colorscale=[[0, 'pink'], [0.28, 'white'], [1.0, 'rgb(70, 130, 180)']],
                    xgap = 1,
                    ygap = 1)
                    ]

    lay = viz.plotly_layout(title = 'Interactive Correlation Matrix',
                            height = 2000, width = 1000, tickangle = 270)
    lay['yaxis']['autorange'] = "reversed"
    lay['yaxis']['showgrid'] = False
    lay['xaxis']['showgrid'] = False
    lay['xaxis']['ticks'] = ''
    lay['yaxis']['ticks'] = ''
    lay['yaxis']['tickfont']['size']=10
    lay['xaxis']['tickfont']['size']=10
    lay['images'] = [dict(
                        source=Image.open(str(pathlib.Path(__file__).parent / 'utils/avataaars.png')),
                        xref="paper", yref="paper",
                        x=0.97, y=0.85, opacity= 0.7,
                        sizex=0.15, sizey=0.15,
                        xanchor="right", yanchor="bottom"#, layer = 'below'
                      )]


    fig = go.Figure(data = data, layout = lay)
    return fig

if __name__ == "__main__":
    main()