##To try and make specific visuals easier, and spark visualisation easier.


##SPARK to visual
##Numeric Variables
def spark_continuous_viz(spark_df, variable, result_name = "Buckets", num_buckets = 10, 
                    description = True, transformed_df = False, plot = True, colour = "lightblue"):
    """To assist in quickly visualising a spark dataframe's continuous variables.
        Returns a full dataframe containing the bucketed variable for further use, 
        printing a quick description of the variable and visual."""
    
    from pyspark.ml.feature import Bucketizer
    from itertools import chain
    from pyspark.sql.functions import create_map, lit, col, min, max, count
    from pyspark.sql.types import StringType
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from IPython.core.display import display as _display
    
    if(isinstance(spark_df.schema[variable].dataType, StringType)==False):
        if(description == True):
            _display(spark_df.describe([variable]).toPandas())
        
        b = spark_df.agg(min(variable).alias('min'), max(variable).alias('max')).toPandas()
        
        splits = [b.iloc[0,0] + x*(b.iloc[0,1]-b.iloc[0,0])/num_buckets for x in range((num_buckets+1))]
        
        
        bucketizer = Bucketizer(splits=splits, inputCol=variable, outputCol="buckets")
        df_buck = bucketizer.setHandleInvalid("keep").transform(spark_df)
        
        names = []
        for i in range(0,len(splits)-1):
            if(splits[i]==np.NINF):
                names.append(np.round(splits[i+1],1))
            elif(splits[i+1]==np.inf):
                names.append(np.round(splits[i],1))
            else:
                names.append(np.round((splits[i]+splits[i+1])/2,1))
        maps = dict(zip(range(0,num_buckets), names))
        
        mapping_expr = create_map([lit(x) for x in chain(*maps.items())])
    
        df = df_buck.withColumn(result_name, mapping_expr[col('buckets')])
        
        grouped = df.groupBy(result_name).agg(count(lit(1)).alias('Count')).toPandas()
        
        if(plot==True):
            sns.set_context('talk')
            sns.set_style('whitegrid')
            
            fig, ax = plt.subplots(figsize=(15, 6))
            g = sns.barplot(ax=ax, x = result_name, y = 'Count', color = colour, data = grouped, order = names) #order = names keeps empty buckets
            g.set_title(result_name)
            
            labels = g.get_xticklabels() # get x labels
            for i,l in enumerate(labels):
                    if num_buckets>=10:
                        if(i%(np.round(num_buckets/10,0)) != 0): labels[i] = '' # skip even labels
            g.set_xticklabels(labels, rotation=90)
            
            plt.show()
        
        if(transformed_df == True):
            return(df)
        else:
            return(None)
    else:
        print('Not a Numeric Variable')
    

##Spark Visuals
##String Variables
def spark_category_viz(spark_df, variable, result_name = 'Category', other_name = 'Other', 
                        top_n = 10, transformed_df = False, 
                        plot = True, palette = "GnBu_d"):
    """To assist in quickly visualising a spark dataframe's categorical variables.
        Will order by frequency and group any variable not in the top_n as 'Other'.
        Returns a full dataframe containing the transformed categorical 
        variable for further use, printing a quick visual.
    """
    
    from pyspark.sql.functions import create_map, col, dense_rank, when, sum
    import seaborn as sns
    import matplotlib.pyplot as plt
    #import matplotlib.ticker as ticker
    from pyspark.sql.types import StringType
    from pyspark.sql.window import Window
    
    
    if(isinstance(spark_df.schema[variable].dataType, StringType)):
        
        
        grouped = spark_df.groupBy(variable).count().\
                    withColumn('rank', dense_rank().over(Window.orderBy(col('count').desc())).alias("rank")).\
                    withColumn(result_name, when(col('rank')<=top_n, col(variable)).otherwise(other_name)).\
                    withColumn('rank2', when(col(result_name)==other_name, top_n+1).otherwise(col('rank'))).\
                    cache()
                    
        plotbase = grouped.groupBy(result_name, 'rank2').agg(sum(col('count')).alias('Count')).toPandas().\
            sort_values('rank2').reset_index()
        
        if(plot == True):
            sns.set_context('talk')
            sns.set_palette(palette)
            sns.set_style('whitegrid')
            
            fig, ax = plt.subplots(figsize=(15, 6))
            g = sns.barplot(ax = ax, x = result_name, y = 'Count', data = plotbase)
            g.set_title(result_name)
            g.set_xticklabels(g.get_xticklabels(), rotation=90)
            for p in g.patches:
                g.text(p.get_x() + p.get_width()/2., (p.get_height()-5), '%d' % int(p.get_height()), 
                        fontsize=12, color='black', ha='center', va='top')
            plt.show()
        
        
        if(transformed_df == True):
            df = spark_df.join(grouped.select(variable, result_name, 'rank'), spark_df[variable]==grouped[variable], how = 'left')
            
            return(df)
        else:
            return(None)
                
    else:
        print("Not a String Variable.")
        
        


        
##Plotly visuals from Pandas df        

def plotly_layout(title = "", xtitle = "", ytitle = "", subtitle = None,
                  height = 500, width = 600, tickangle = 0):
                      
    """
    Branded plotly layout
    """
    
    import plotly
    import plotly.graph_objs as go
    from PIL import Image
    import pathlib
    
                          
    if(subtitle is not None):
        ttl = '{}<br><i><span style="font-size:14pt">{}</span></i>'.format(title, subtitle)
    else:
        ttl = '<br>{}<br>'.format(title)
        
    layout = go.Layout(
                bargap = 0.3,
                hovermode = 'x',
                height = height,
                width = width,
                paper_bgcolor='white',
                plot_bgcolor='white',

                title = dict(text = ttl, ##MAKE PARAM
                         font = dict(
                            family='Arial, sans-serif',
                            size=26,
                            color='black'
                        ),
                        xanchor = 'center',
                        x = 0.5
                        ),
                
                legend=dict(
                            font=dict(
                                family='Arial, sans-serif',
                                size=14,
                                color='#000'
                            )),
                
                xaxis=go.layout.XAxis(
                    title = xtitle, ##MAKE PARAM
                    titlefont = dict(
                            family='Arial, sans-serif',
                            size=14,
                            color='black'
                        ),
                    tickfont=dict(
                            family='Arial, sans-serif',
                            size=14,
                            color='black'
                        ),
                    hoverformat = '.1f',
                    tickangle = tickangle,
                    # showgrid=True,
                    # gridwidth=1, 
                    # gridcolor='LightGrey',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGrey',
                    # showline=True,
                    automargin= True,
                    mirror='ticks'),

                yaxis=go.layout.YAxis(
                    title = ytitle, ##MAKE PARAM
                    titlefont = dict(
                            family='Arial, sans-serif',
                            size=14,
                            color='black'
                        ),
                    tickfont=dict(
                            family='Arial, sans-serif',
                            size=14,
                            color='black'
                        ),
                    hoverformat = '.1f',
                    automargin= True,
                    showgrid=True,
                    gridwidth=1, 
                    gridcolor='LightGrey',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='LightGrey',
                    # showline=True,
                    mirror='ticks')
        ,

                images=[dict(
                        source=Image.open(str(pathlib.Path(__file__).parent / 'avataaars.png')),
                        xref="paper", yref="paper",
                        x=0.97, y=0.03, opacity= 0.7,
                        sizex=0.15, sizey=0.15,
                        xanchor="right", yanchor="bottom"#, layer = 'below'
                      )]
    )
    
    return layout


def plotlys(df, x, 
               y = None, 
               kind = None,
               
               group = None,
               
               #Scatter
               size = None,
               colour_reverse = False,
               
               #Bar
               stacked = False,
               annotate = False, 
               ordered = True,
               
               #Layout
               title = "", xtitle = None, ytitle = None, subtitle = None, 
               height = 500, width = 600, tickangle = 0
              ):
    """
    To have branded plots created in a single line, with simple grouping and formatting.
    
    Accepts Pandas DF - ensure it is pre aggregated for grouped visuals.
    Select one of 'Hist' (Histogram), 'Bar', or 'Scat' (Scatter) to produce a suitable plot.
    """
    import plotly
    import plotly.graph_objs as go
    import numpy as np
    
    
    # plotly.init_notebook_mode(connected=False)

    
    
    try:
        assert kind in ['Hist', 'Bar', 'Scat']
    except AssertionError:
        print('Select one of Hist, Bar, Scat Graph Type')
        return(None)
        
    ##Variable & Layout Setup
    colours = ['midnightblue','darkmagenta', 'seagreen', 'firebrick','#f4bc61', 
           '#546886', '#4ba4c6','mediumseagreen','lightcoral']   
    
    colourscale = [[0, '#f54411'], [0.5, '#faf8ee'], [1.0, '#00ac70']]
    if(colour_reverse == True):
        colourscale = [[0, '#00ac70'], [0.5, '#faf8ee'], [1.0, '#f54411']]
    
    if((title == "") & (y is not None)):
        title = "{} vs {}".format(x.replace('_', ' ').title(), 
                                       y.replace('_', ' ').title())
    elif((title == "")):
        title = "{} Distribution".format(x.replace('_', ' ').title())
    
    if(xtitle is None):
        xtitle = x.replace('_', ' ').title() 
        
    if((ytitle is None) & (y is not None)):
        ytitle = y.replace('_', ' ').title() 
    elif(ytitle is None):
        ytitle = 'Volume'
    
    layout = plotly_layout(title, xtitle, ytitle, subtitle,
                           height, width, tickangle)

    
    
    ##Histogram
    if(kind == 'Hist'):
        
        if(annotate == True):
            print('Annotations not available for Histograms')
        
        if(group is None):
            base = [go.Histogram(
                    x = df[x],
                    marker = dict(
                                    color='midnightblue',
                                    line=dict(
                                        color='white',
                                        width=1
                                    ))) 
                   ]
        else:
            coloursIdx = dict(zip(df[group].unique(), colours))
            
            base = [go.Histogram(
                        opacity = 0.5,
                        x = df[df[group]==val][x],
                        name = val, 
                        marker = dict(color=coloursIdx.get(val),
                                        line=dict(
                                            color='white',
                                            width=1
                                        )
                                        )) for val in df[group].unique()
                   ]
            layout.update({'barmode':'overlay'})
    
    
    ##Bar
    elif(kind == 'Bar'):
        if(group is None):
            if(annotate == True):
                    annotate = 'auto'
                    t = np.round(df[y])
            else:
                    annotate = 'none'
                    t = None
            
            if(ordered == True):
                df.sort_values(by=y,
                         axis=0,    
                         ascending=False,  
                         inplace = True)
            
            base = [
                go.Bar(
                    x = df[x],
                    y = df[y],
                    marker = dict(
                                color='midnightblue',
                                line=dict(
                                    color='white',
                                    width=1,
                                )),
                    text = t,
                    textposition = annotate,
                    textfont=dict(
                        family='Arial, sans-serif',
                        size=14,
                        color='darkgrey'
                    )
                )
            ]
        else:
            
            if(annotate == True):
                        annotate = 'auto'
            else:
                        annotate = 'none'
                        
            def annotation(annotate, val):
                if(annotate == 'auto'):
                        t = np.round(df[df[group]==val][y])
                else:
                        t = None
                return t
            
            if(stacked == True):
                layout.update({'barmode':'stack'})
                
            coloursIdx = dict(zip(df[group].unique(), colours))
            #cols      = df[group].map(coloursIdx)
            
            if(ordered == True):
                df.sort_values(by=y,
                         axis=0,    
                         ascending=False,  
                         inplace = True)
            
            base =  [
                go.Bar(
                    x=df[df[group]==val][x],
                    y = df[df[group]==val][y],
                    name = val,
                    marker = dict(color = coloursIdx.get(val)),
                    text = annotation(annotate, val),
                    textposition = annotate,
                    textfont=dict(
                            family='Arial, sans-serif',
                            size=14,
                            color='darkgrey'
                        )
        
        
                    ) for val in df[group].unique()
                ] 
    
    
    ##Scatter
    elif(kind == 'Scat'):
        
        if(annotate == True):
            print('Annotations not available for Scatter Plots')
        
        if(group is None):
            if(size is None):
                
                base = [
                    go.Scatter(
                        opacity = 0.75,
                        x = df[x],
                        y = df[y],
                        marker=dict(
                                    color='midnightblue',
                                    line=dict(
                                        color='white',
                                        width=1,
                                    ),
                                    size = 16),
                        mode = 'markers'
                      ) 
                    ]
                
            else:  
                base = [
                        go.Scatter(
                            x = df[x],
                            y = df[y],
                            marker=dict(
                                color='midnightblue',
                                line=dict(
                                        color='white',
                                        width=1,
                                    ),
                                size = df[size],
                                sizeref=2.*np.max(df[size])/(10.**2),
                                sizemin=10
                                        ),
                            mode = 'markers'
                          ) 
                        ]
        
        else:
            if(size is not None):
                if(df[group].dtype == np.float64 or df[group].dtype == np.int64):

                    base = [
                        go.Scatter(
                            x = df[x],
                            y = df[y],
                            text = df[group],
                            marker=dict(
                                color=df[group],
                                colorbar=dict(
                                        title=group.replace('_', ' ').title()
                                            ),
                                colorscale = colourscale,
                                size = df[size],
                                sizeref=2.*np.max(df[size])/(10.**2),
                                sizemin=10
                                        ),
                            mode = 'markers'
                          ) 
                        ]

                else:

                    coloursIdx = dict(zip(df[group].unique(), colours))

                    base = [
                            go.Scatter(
                                x = df[df[group]==val][x],
                                y = df[df[group]==val][y],
                                name = val,
                                marker = dict(size = df[df[group]==val][size],
                                              sizeref=2.*np.max(df[size])/(10.**2),
                                              sizemin=10,
                                              color = coloursIdx.get(val)),
                                mode = 'markers'
                        ) for val in df[group].unique()

                        ]

            else:
                if(df[group].dtype == np.float64 or df[group].dtype == np.int64):

                    base = [
                        go.Scatter(opacity = 0.75,
                            x = df[x],
                            y = df[y],
                            text = df[group],
                            marker=dict(
                                color=df[group],
                                colorbar=dict(
                                        title=group.replace('_', ' ').title()
                                            ),
                                colorscale = colourscale,
                                line=dict(
                                        color='white',
                                        width=1,
                                    ),
                                size = 16
                                        ),
                            mode = 'markers'
                          ) 
                        ]

                else:

                    coloursIdx = dict(zip(df[group].unique(), colours))

                    base = [
                            go.Scatter(
                                opacity = 0.75,
                                x = df[df[group]==val][x],
                                y = df[df[group]==val][y],
                                name = val,
                                marker = dict(size = 16,
                                              color = coloursIdx.get(val),
                                              line=dict(
                                                        color='white',
                                                        width=1,
                                                    )),
                                mode = 'markers'
                        ) for val in df[group].unique()

                        ]
        
    fig = go.Figure(data = base, layout = layout)
        
    return fig