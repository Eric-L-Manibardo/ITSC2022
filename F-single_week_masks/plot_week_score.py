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

sns.despine(left=True)
sns.set(style="whitegrid")

val_weeks = pd.read_csv('val_weeks_score.csv', index_col=0).T
test_weeks = pd.read_csv('test_weeks_score.csv', index_col=0).T
z=0

for i in range(len(val_weeks)):
    plt.plot(range(1,53),val_weeks.iloc[i],c='#ff7f0e')
    plt.plot(range(1,53),test_weeks.iloc[i],c='#1f77b4')

plt.xlabel('Sensed week of the year')
plt.ylabel('RMSE')
plt.legend(['Validation error',
            'Test error'
    
    ])

plt.xlim([1,52])
plt.ylim([0,570])

# for i in range(10):
#     fig, ax = plt.subplots(3, 3, sharex=False, sharey=True)

#     for row in ax:
#         for col in row:
#             col.plot(val_weeks.iloc[z])
#             col.plot(test_weeks.iloc[z])
                
                
            
#             z=z+1
#     ax[0][2].legend(['Validation (no target data)','Test (target data)'])
#     for k in range(3):
#         ax[k][0].set_ylabel('Score per week')
#     for k in range(3):
#         ax[2][k].set_xlabel('Training weeks')
#     fig.suptitle('Validation and test score for mask of size 1 (one mask for every week). ATRs from ' +str(i*9)+' to '+str(i*9+8))
    
    
        
        
        
        
        
        
        
        
        

