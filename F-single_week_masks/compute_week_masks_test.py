#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:33:29 2022

@author: eric
sacamos un score de test para cada semana del año
guardamos a csv y en otro script ploteo la curva
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

def build_dataset(tr1,tr2,tr3,target):
    # contruyo dataset de entrenamiento {t-5,...,t-1} y target en {t}
    x1 = tr1[0:-5].squeeze()
    x2 = tr1[1:-4].squeeze()
    x3 = tr1[2:-3].squeeze()
    x4 = tr1[3:-2].squeeze()
    x5 = tr1[4:-1].squeeze()
    x6 = tr2[0:-5].squeeze()
    x7 = tr2[1:-4].squeeze()
    x8 = tr2[2:-3].squeeze()
    x9 = tr2[3:-2].squeeze()
    x10 = tr2[4:-1].squeeze()
    x11 = tr3[0:-5].squeeze()
    x12 = tr3[1:-4].squeeze()
    x13 = tr3[2:-3].squeeze()
    x14 = tr3[3:-2].squeeze()
    x15 = tr3[4:-1].squeeze()
    y = target[5:].squeeze()
    
    # creamos el dataframe
    df= pd.DataFrame({"x1": x1, "x2": x2,"x3": x3,"x4": x4,"x5": x5,
                     "x6": x6,"x7": x7,"x8": x8,"x9": x9,"x10": x10,
                     "x11": x11,"x12": x12,"x13": x13,"x14": x14,"x15": x15,
                     'y':y})
    return df

def train_model_mask(trainA,trainB,trainC,target,mask):
    '''
    entreno un modelo con 3 espiras de acuerdo a mascara optimizada
    '''
    trainA = selected_weeks(trainA, mask)
    trainB = selected_weeks(trainB, mask)
    trainC = selected_weeks(trainC, mask)
    target = selected_weeks(target, mask)
    
    df_train = build_dataset(trainA, trainB,trainC, target)
    
    y_train = df_train['y'].values
    #guardamos el scaler para test
    scaler = StandardScaler().fit(df_train.iloc[:,:-1].values)
    X_train = scaler.transform(df_train.iloc[:,:-1].values)
    
    model = ExtraTreesRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)   

    return model, scaler

def train_model_full(trainA,trainB,trainC,target):
    '''
    entreno un modelo con 3 espiras con todo 2018
    '''

    #lo pasamos como array
    trainA = trainA['flow'].values.flatten()
    trainB = trainB['flow'].values.flatten()
    trainC = trainC['flow'].values.flatten()
    target = target['flow'].values.flatten()
    df_train = build_dataset(trainA, trainB,trainC, target)
    
    y_train = df_train['y'].values
    #guardamos el scaler para test
    scaler = StandardScaler().fit(df_train.iloc[:,:-1].values)
    X_train = scaler.transform(df_train.iloc[:,:-1].values)
    
    model = ExtraTreesRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)   

    return model, scaler

def test_model(inputA,inputB,inputC, test, model, scaler):
    '''
    valido el modelo con el año completo 2019
    '''
    # extraigo como array
    inputA = inputA['flow'].values.flatten()
    inputB = inputB['flow'].values.flatten()
    inputC = inputC['flow'].values.flatten()
    test = test['flow'].values.flatten()
    
    df_test = build_dataset(inputA, inputB,inputC, test)
    
    y_test = df_test['y'].values
    X_test = scaler.transform(df_test.iloc[:,:-1].values)
    
    y_pred = model.predict(X_test)
    
    return sqrt(mean_squared_error(y_test, y_pred))

# =============================================================================
# START HERE
# =============================================================================

d = {}
for z in range(55):
    print(z)
    # LOAD DATA
    selected_loops = pd.read_csv('selected_loops.csv', index_col=0)
    selec=z
    donor1 = pd.read_csv('B-colombia_LIMPIO_2018_2019/'+str(selected_loops.iloc[selec][1])+'.csv',index_col=0)
    donor2 = pd.read_csv('B-colombia_LIMPIO_2018_2019/'+str(selected_loops.iloc[selec][2])+'.csv',index_col=0)
    donor3 = pd.read_csv('B-colombia_LIMPIO_2018_2019/'+str(selected_loops.iloc[selec][3])+'.csv',index_col=0)
    target = pd.read_csv('B-colombia_LIMPIO_2018_2019/'+str(selected_loops.iloc[selec][0])+'.csv',index_col=0)
    
    train1, test1 = split_train_test(donor1)
    train2, test2 = split_train_test(donor2)
    train3, test3 = split_train_test(donor3)
    train_target, test_target = split_train_test(target)
    
    # computo el score para cada semana
    n_weeks=52
    score_week = list()
    for i in range(n_weeks):
        mask=np.zeros(52,dtype=bool)
        #asignamos la semana pertinente
        mask[i]=True
        model_mask, scaler_mask = train_model_mask(train1,train2,train3,train_target,mask)
        rmse_mask = test_model(test1,test2,test3,test_target,model_mask,scaler_mask)
        score_week.append(rmse_mask)
        
    d[selec] = np.array(score_week)

    
df_save = pd.DataFrame.from_dict(d)
df_save.to_csv('test_weeks_score.csv')










