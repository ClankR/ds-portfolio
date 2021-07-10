from utils import viz

import os
pre = os.getcwd()

import pandas as pd
df = pd.read_csv(pre+'/Data/cell2celltrain.csv')

base = df.groupby('ServiceArea').agg({'MonthlyRevenue':'mean', 'MonthlyMinutes':'mean'}).reset_index()
fig = viz.plotlys(base, x = 'MonthlyRevenue', y = 'MonthlyMinutes', group = 'ServiceArea', kind = 'Scat')
import plotly
plotly.iplot(fig, config = [('displaylogo', False)])