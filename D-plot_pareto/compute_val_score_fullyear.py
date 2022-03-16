'''
sacamos un score de validacion para cada semana del ano completo
guardamos a csv y en otro script ploteo la curva
'''


#other imports
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


# =============================================================================
# other functions
# =============================================================================
def print_function_values_to_csv(solutions, filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]   

    a=list()
    for s in solutions:
        a.append(s.objectives)
    with open(filename+'.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerows(a)
   
    
def print_variables_to_csv(solutions, filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    if type(solutions) is not list:
        solutions = [solutions]   
  
    a=list()
    for s in solutions:
        a.append(s.variables)
    a=np.squeeze(a)
    with open(filename+'.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerows(a)
    


    
            
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
    if len(a)==1:
        return np.array(a)
        # print('ONE WEEK ONLY')
    else:
        #no se por qué me lo devuelve como array de objs, asi lo deja bien
        return np.concatenate(np.array(a)).astype(None)

def build_dataset(tr1,tr2,target):
    # en caso de maskara de 1 semana
    if tr1.shape[0]==1:
        x1 = tr1[0][0:-5]
        x2 = tr1[0][1:-4]
        x3 = tr1[0][2:-3]
        x4 = tr1[0][3:-2]
        x5 = tr1[0][4:-1]
        x6 = tr2[0][0:-5]
        x7 = tr2[0][1:-4]
        x8 = tr2[0][2:-3]
        x9 = tr2[0][3:-2]
        x10 = tr2[0][4:-1]
        y = target[0][5:]
    else:
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
        y = target[5:].squeeze()
    
    # creamos el dataframe
    df= pd.DataFrame({"x1": x1, "x2": x2,"x3": x3,"x4": x4,"x5": x5,
                     "x6": x6,"x7": x7,"x8": x8,"x9": x9,"x10": x10,
                     'y':y})
    return df

def train_model(trainA,trainB,target,mask):
    '''
    entreno un modelo con dos espiras de acuerdo a mascara 2016
    '''
    trainA = selected_weeks(trainA, mask)
    trainB = selected_weeks(trainB, mask)
    target = selected_weeks(target, mask)
    
    df_train = build_dataset(trainA, trainB, target)
    
    y_train = df_train['y'].values
    #guardamos el scaler para test
    scaler = StandardScaler().fit(df_train.iloc[:,:-1].values)
    X_train = scaler.transform(df_train.iloc[:,:-1].values)
    
    model = ExtraTreesRegressor(n_estimators=100)
    
    model.fit(X_train, y_train)   

    return model, scaler

def validate_model(inputA,inputB, val, model, scaler):
    '''
    valido el modelo con el año completo 2017
    '''
    # extraigo como array
    inputA = inputA['flow'].values.flatten()
    inputB = inputB['flow'].values.flatten()
    val = val['flow'].values.flatten()
    
    df_val = build_dataset(inputA, inputB, val)
    
    y_val = df_val['y'].values
    X_val = scaler.transform(df_val.iloc[:,:-1].values)
    
    y_pred = model.predict(X_val)
    
    return mean_squared_error(y_val, y_pred)

def compute_mean_score(train1, train2,train3,val1,val2,val3, mask):
    '''
    returns mean validation score from the 3 trained models, according to the mask
    '''
    # Training models
    model1,scaler1 = train_model(train2,train3,train1,mask)
    model2,scaler2 = train_model(train3,train1,train2,mask)
    model3,scaler3 = train_model(train1,train2,train3,mask)
    
    # Validating models
    mse1 = validate_model(val2,val3, val1, model1, scaler1)
    mse2 = validate_model(val3,val1, val2, model2, scaler2)
    mse3 = validate_model(val1,val2, val3, model3, scaler3)
    
    return sqrt(np.mean([mse1,mse2,mse3]))
    
def split_train_val(df):
    '''
    separa 2016 (train) de 2017 (validation)
    '''
    #2016 es bisiesto -> 96 samples * 366 days = 35136
    split = 35136
    train = df.iloc[:split]
    val = df.iloc[split:]
    
    return train, val        

def build_mask(mask_size):
    #number of positives in mask
    n_positives = random.randint(1,mask_size)
    #relleno de Falses
    mask = [False]*mask_size
    cont=0
    while (cont < n_positives):
        #posicion del True
        pos = random.randint(0,mask_size-1)
        if not mask[pos]:
            mask[pos] = True
            cont = cont+1
    
    return mask    

 


    
# =============================================================================
# START HERE
# =============================================================================
d={}
selec=-1
while(selec<54):
    # LOAD DATA
    selected_loops = pd.read_csv('selected_loops.csv', index_col=0)
    selec=selec+1
    donor1 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][1])+'.csv',index_col=0)
    donor2 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][2])+'.csv',index_col=0)
    donor3 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][3])+'.csv',index_col=0)
    
    train1, val1 = split_train_val(donor1)
    train2, val2 = split_train_val(donor2)
    train3, val3 = split_train_val(donor3)    
    
    
    # computo el score para cada semana
    n_weeks=52
    score_week = list()

    mask=np.ones(52,dtype=bool)
    #asignamos la semana pertinente
    score = compute_mean_score(train1, train2,train3,val1,val2,val3, mask)

        
    d[selec] = score
    
   
df_save = pd.DataFrame.from_dict([d])
df_save.to_csv('val_fullyear_score.csv')


