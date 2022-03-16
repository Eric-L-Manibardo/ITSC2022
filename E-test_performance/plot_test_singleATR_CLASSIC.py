#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:33:29 2022

@author: eric

usamos el approach de modelaje clásico
"""

import random
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import csv
from matplotlib import pyplot as plt


def split_train_test(df):
    '''
    separa 2016 (train) de 2017 (validation)
    '''
    #2018 NO es bisiesto -> 96 samples * 365 days = 35040
    split = 96*365
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    return train, test 

def selected_weeks(df,mask):
    '''
    filtra las semanas indicadas en la máscara
    devuelve en formato array 1D
    '''
    
    week=96*7 #samples in a week
    X = df['flow'].values
    a=list()
    i=0
    for m in mask:
        if m==True:
            if i==week*51:#ultima semana, añadir 2 dias extra por ser bisiesto
                a.append(X[i+week:])
            else:
                a.append(X[i:i+week])
        i=i+week
        
    #no se por qué me lo devuelve como array de objs, asi lo deja bien
    return np.concatenate(np.array(a)).astype(None)

def build_dataset(target):
    # contruyo dataset de entrenamiento {t-5,...,t-1} y target en {t}
    x1 = target[0:-5].squeeze()
    x2 = target[1:-4].squeeze()
    x3 = target[2:-3].squeeze()
    x4 = target[3:-2].squeeze()
    x5 = target[4:-1].squeeze()
    y = target[5:].squeeze()
    
    # creamos el dataframe
    df= pd.DataFrame({"x1": x1, "x2": x2,"x3": x3,"x4": x4,"x5": x5,
                     'y':y})
    return df

def train_model_mask(target,mask):
    '''
    entreno un modelo con 3 espiras de acuerdo a mascara optimizada
    '''

    target = selected_weeks(target, mask)
    
    df_train = build_dataset(target)
    
    y_train = df_train['y'].values
    #guardamos el scaler para test
    scaler = StandardScaler().fit(df_train.iloc[:,:-1].values)
    X_train = scaler.transform(df_train.iloc[:,:-1].values)
    
    model = ExtraTreesRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)   

    return model, scaler

def train_model_full(target):
    '''
    entreno un modelo con 3 espiras con todo 2018
    '''

    #lo pasamos como array
    target = target['flow'].values.flatten()
    df_train = build_dataset(target)
    
    y_train = df_train['y'].values
    #guardamos el scaler para test
    scaler = StandardScaler().fit(df_train.iloc[:,:-1].values)
    X_train = scaler.transform(df_train.iloc[:,:-1].values)
    
    model = ExtraTreesRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)   

    return model, scaler

def test_model(test, model, scaler):
    '''
    valido el modelo con el año completo 2019
    '''
    # extraigo como array

    test = test['flow'].values.flatten()
    
    df_test = build_dataset(test)
    
    y_test = df_test['y'].values
    X_test = scaler.transform(df_test.iloc[:,:-1].values)
    
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

# =============================================================================
# START HERE
# =============================================================================
extra_loops = [4553,4693,3947]
final_mask, final_full = list(), list()

# LOAD DATA
selected_loops = pd.read_csv('selected_loops.csv', index_col=0)
selec=34
target = pd.read_csv('B-colombia_LIMPIO_2018_2019/'+str(selected_loops.iloc[selec][0])+'.csv',index_col=0)



#split train test

train_target, test_target = split_train_test(target)




res = pd.read_csv('ND_results/ND_results_'+str(selec)+'.csv', index_col=False, header=None)
mask = pd.read_csv('ND_variables/ND_variables_'+str(selec)+'.csv', index_col=False, header=None)
#usamos la máscara más pequeña
idx_min = res[0].idxmin()



#cargamos las máscaras
mask_min = mask.iloc[idx_min].values



# de acuerdo a mascara
model_mask_min, scaler_mask_min = train_model_mask(train_target,mask_min)


# con todos los datos
model_full, scaler_full = train_model_full(train_target)

_, y_pred_min = test_model(test_target,model_mask_min,scaler_mask_min)

y_test, y_pred_full = test_model(test_target,model_full,scaler_full)



fig = plt.figure()
#Sobran las 5 primeras muestras
# plt.plot(test1.iloc[5:].values, linestyle='dotted')
# plt.plot(test2.iloc[5:].values, linestyle='dashed')
# plt.plot(test3.iloc[5:].values, linestyle='dashdot')
plt.plot(y_pred_min)

plt.plot(y_pred_full)
plt.plot(y_test, color='k')

plt.ylabel('Flow (vehicles per 15 minutes)')
plt.xlabel('Date')
plt.title('Traffic forecasting fit over full and partial 2018. Test over full 2019')
# plt.legend(['Weeks for training: '+str(int(res.loc[idx_min][0])),
#             'Weeks for training: '+str(int(res.loc[idx_medio][0])),
#             'Weeks for training: '+str(int(res.loc[idx_max][0])),
#             'Weeks for training: '+str(52),
#             'Real data'
#             ])
plt.legend([
    # 'Donor 1',
    #         'Donor 2',
    #         'Donor 3',
            'Single week for training',
            'All year for training',
            'Real data'
            ])

eje_x = pd.date_range(start='2019/01/01',end='2019/12/31 23:45:00',freq='15MIN' )
eje_x = eje_x.strftime('%Y/%m/%d')
frec=96
plt.xticks(range(len(eje_x))[::frec], eje_x[::frec], size='small', rotation=45, horizontalalignment='center')


error_full =sqrt(mean_squared_error(y_test,y_pred_full))
error_min = sqrt(mean_squared_error(y_test,y_pred_min))
print('Test RMSE ATR '+str(selec)+'| Full 2018 year (baseline): '+str(error_full)+' and mask ('+str(min(res[0]))+'): ' +str(error_min))






