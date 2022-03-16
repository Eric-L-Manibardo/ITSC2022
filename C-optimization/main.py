

# carpeta de archivos en local
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.problem import Problem
from jmetal.core.solution import BinarySolution
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file

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
        print('ONE WEEK ONLY')
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
    
    return sqrt(mean_squared_error(y_val, y_pred))

def compute_mean_score(train1, train2,train3,val1,val2,val3, mask):
    '''
    returns mean validation score from the 3 trained models, according to the mask
    '''
    # Training models
    model1,scaler1 = train_model(train2,train3,train1,mask)
    model2,scaler2 = train_model(train3,train1,train2,mask)
    model3,scaler3 = train_model(train1,train2,train3,mask)
    
    # Validating models
    rmse1 = validate_model(val2,val3, val1, model1, scaler1)
    rmse2 = validate_model(val3,val1, val2, model2, scaler2)
    rmse3 = validate_model(val1,val2, val3, model3, scaler3)
    
    return np.mean([rmse1,rmse2,rmse3])
    
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
# Class myProblem
# =============================================================================

class myProblem(Problem):
    def __init__(self, num_variables,mask_size,train1,train2,train3, val1,val2,val3):
        
        super(myProblem, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = num_variables
        self.number_of_constraints = 0
        self.mask_size = mask_size
        self.obj_directions = [self.MINIMIZE,self.MINIMIZE]
        self.obj_labels = ['Mask size','Mean MSE']
        self.train1 = train1
        self.train2 = train2
        self.train3 = train3
        self.val1 = val1
        self.val2 = val2
        self.val3 = val3

    
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        ''' objetivos a cumplir'''
        
        # OBJ 1: reducir 1s en la maskara de 52 valores
        solution.objectives[0] = sum([1 if s else 0 for s in solution.variables[0]])        
        
        # OBJ 2: reducir el MSE de las predicciones de tráfico
        '''
        Para ello obtengo 3 MSE scores de la siguiente función y hago el promedio
        '''
        score = compute_mean_score(train1, train2,train3,val1,val2,val3, solution.variables[0])        
        
        solution.objectives[1] = score
        
        print('Mask: '+str(solution.objectives[0])+'/52. Mean RMSE: '+str(score))
        return solution
    

    


    def create_solution(self) -> BinarySolution:
        #aqui se generan las distintas mascaras a probar
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        # for i in range(self.number_of_variables):
        #     new_solution.variables[i] = True if random.randint(0, 1)==1 else False
        # new_solution.variables[0] = [True if random.randint(0, 1)==1 else False for _ in range(self.mask_size)]
            
        new_solution.variables[0] = build_mask(mask_size)
            
        return new_solution
    
    def get_name(self) -> str:
        return "ITSC_2022_optimization"
   


    
# =============================================================================
# START HERE
# =============================================================================
if __name__ == '__main__':
    selec=34
    while(selec<55):
        
	# LOAD DATA
        selected_loops = pd.read_csv('selected_loops.csv', index_col=0)
        
        donor1 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][1])+'.csv',index_col=0)
        donor2 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][2])+'.csv',index_col=0)
        donor3 = pd.read_csv('B-colombia_LIMPIO_2016_2017_v2/'+str(selected_loops.iloc[selec][3])+'.csv',index_col=0)
        
        train1, val1 = split_train_val(donor1)
        train2, val2 = split_train_val(donor2)
        train3, val3 = split_train_val(donor3)    
        
        # 1 variable, la propia máscara de 52 semanas
        mask_size = 52
        # son 52 variables booleanas
        NUMBER_VARIABLES = 1
        max_evaluations = 500
        problem = myProblem(NUMBER_VARIABLES,mask_size,train1,train2,train3, val1,val2,val3)
        
        progress_bar = ProgressBarObserver(max=max_evaluations)
        
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=BitFlipMutation(probability=1.0 / NUMBER_VARIABLES),
            crossover=SPXCrossover(probability=0.9),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
        
        algorithm.observable.register(progress_bar)
        algorithm.run()

        
        # Nos quedamos solo con aquellas soluciones en el Pareto Front: aquellas que no son dominadas por otras
        front = get_non_dominated_solutions(algorithm.get_result())      
        print_function_values_to_csv(front, 'aND_results/ND_results_'+str(selec))
        print_variables_to_csv(front, 'aND_variables/ND_variables_'+str(selec))
        
        # todos los resultados
        front= algorithm.get_result()
        print_function_values_to_csv(front, 'afull_results/full_results_'+str(selec))
        print_variables_to_csv(front, 'afull_variables/full_variables_'+str(selec))
    
        print(f'Algorithm: {algorithm.get_name()}')
        print(f'Problem: {problem.get_name()}')
        print(f'Computing time: {algorithm.total_computing_time}')
        selec=selec+1
        break
