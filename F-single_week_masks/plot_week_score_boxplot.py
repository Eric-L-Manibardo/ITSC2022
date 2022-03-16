#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:56:27 2022

@author: eric

dibujamos las soluciones no dominantes

NORMALIZAMOS EJE X CON EL RMSE DE USAR 52 SEMANAS
"""

import csv
import pandas as pd
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.titlesize'] = 20
sns.set_style("whitegrid")



val_weeks = pd.read_csv('val_weeks_score.csv', index_col=0).T
test_weeks = pd.read_csv('test_weeks_score.csv', index_col=0).T


week, error, hue = list(), list(), list()

# rellenamos con validation
for i in range(val_weeks.shape[0]): # 55 ATRs
    for j in range(val_weeks.shape[1]): # 52 semanas
        week.append(j+1)
        error.append(val_weeks.iloc[i][j])
        hue.append('Validation')
        
# rellenamos con test
for i in range(test_weeks.shape[0]): # 55 ATRs
    for j in range(test_weeks.shape[1]): # 52 semanas
        week.append(j+1)
        error.append(test_weeks.iloc[i][j])
        hue.append('Test')
        
        
df_draw = pd.DataFrame({'Sensed week of the year':week, 'RMSE':error, 'Committed error':hue})

flierprops = dict(marker='x', markerfacecolor='None', markersize=4,  markeredgecolor='grey')


ax = sns.boxplot(data=df_draw, x='Sensed week of the year', y='RMSE', hue='Committed error',
                 flierprops=flierprops, dodge=True,width=0.6)

ax.set_ylim([0,570])
#frecuencia del eje x
ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter()) 
        
        
        
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    