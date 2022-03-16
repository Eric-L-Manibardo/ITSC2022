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
from matplotlib import pyplot as plt
# import pareto

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


best = pd.read_csv('val_fullyear_score.csv', index_col=0)
z=0
#espira a plotear
selec = 34

df = pd.read_csv('ND_results/ND_results_'+str(selec)+'.csv', index_col=False, header=None)
df = df.rename(columns={0: 'Mask_size', 1: 'RMSE'})

#normalizamos eje X
df['RMSE'] = df['RMSE'].div(best.loc[0][selec]/100)

df_full = pd.read_csv('full_results/full_results_'+str(selec)+'.csv', index_col=False, header=None)
df_full = df_full.rename(columns={0: 'Mask_size', 1: 'RMSE'})

#normalizamos eje X
df_full['RMSE'] = df_full['RMSE'].div(best.loc[0][selec]/100)
z=z+1
    
fig = plt.figure()
    
plt.scatter(data=df_full,x='RMSE', y='Mask_size', color='tab:blue') 
plt.scatter(data=df,x='RMSE', y='Mask_size', color='tab:orange') 
#ordenamos por MSE para que sea linea continua
df = df.sort_values(by=['RMSE'])
plt.plot(df['RMSE'].values, df['Mask_size'].values,color='tab:orange') 

plt.legend(['Non-dominant','All results'])

plt.ylabel('Training weeks')

plt.xlabel('Validation score (nRMSE  \%)')
# plt.title('Selected number of training weeks for ATRs from ' +str(i*9)+' to '+str(i*9+8))

    
        
        
        
        
        
        
        
        
        

