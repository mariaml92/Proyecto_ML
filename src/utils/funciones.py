#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long

#-------------------------------------------------------------------------------

'''
Este fichero contiene las funciones utilizadas en el proyecto de ML sobre la predección del riesgo de sufrir un accidente cerebro-cardiovascular.
https://github.com/mariaml92
'''

#-------------------------------------------------------------------------------
#LIBRERÍAS UTILIZADAS
import numpy as np
import math
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
sns.set(color_codes=True)

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import xgboost
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score
import xgboost
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier

from collections import Counter

import pickle

import time as tm

import warnings
warnings.simplefilter('ignore')

#-------------------------------------------------------------------------------

def cargar_datos(enlace):
    '''
    Esta función carga el dataset que queremos utilizar

    Parámetros:
    - enlace:Enlace con la ruta al archivo

    Devuelve:
    -DataFrame con los datos
    '''

    return pd.read_csv(enlace)

#-------------------------------------------------------------------------------

def procesar_datos(df):
    '''
    Esta función realiza todo el procesado de los datos previos al entrenamiento
    '''

    # Eliminar outliers
    df=eliminar_outliers(df, 'gender', 'Other')
    
    # Tranformar de variables categóricas
    df = df_encoding(df)

    # Guardar el dataset procesado
    dir='./data/processed_data'
    nombre='dataset_procesado.csv'
    guardar_dataset(df, dir, nombre)

    #Dividir en train y test
    target='stroke'
    X, y, X_train, X_test, y_train, y_test=dividir_train_test(df, target)

    # Imputar missings
    X_train=imputar_media(X_train, X_train, 'bmi')
    X_test=imputar_media(X_train, X_test, 'bmi')

    # Feature reduction
    drop_column_list=['id', 'Residence_type']
    feature_reduction(X_train, drop_column_list)
    feature_reduction(X_test, drop_column_list)
    
    # Escalar
    X_train_scaled=escalar_datos(X_train,X_train)
    X_test_scaled=escalar_datos(X_train,X_test)

    return df, X, y, X_train_scaled, X_test_scaled, y_train, y_test

#-------------------------------------------------------------------------------

def eliminar_outliers(df, columna, valor):
    '''
    Esta función elimina los outliers con el valor especificado
    
    Parámetros:
    - df: DataFrame de que queremos eliminar los outliers
    - columna:Columna del df donde está el valor a eliminar
    - valor:Valor de la columna que queremos eliminar

    Devuelve:
    -DataFrame con outliers eliminados
    '''

    indice = df[ df[columna] == valor ].index
    df.drop(indice, inplace = True)

    return df

#-------------------------------------------------------------------------------

def data_report(df):

    '''
    Esta función describe los campos de un dataframe: nombres, tipo, missings, valores únicos 
    y el porcentaje de valores únicos respecto al total de valores de esa variable (Cardin)

    Parámetros:
    - df: DataFrame del que queremos obtener la información

    Devuelve:
    - concatenado.T: tabla con nombres, tipo, missings, valores únicos y Cardin del DataFrame

    '''
    
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

#-------------------------------------------------------------------------------

def df_encoding(df):
    '''
    Esta función transforma las variables categóricas a numéricas del DatraFrame con datos de riesgo de accidentes cerebrovasculares

    Parámetros:
    - df: DataFrame del que quiero transformar las variables categóricas

    Devuelve:
    - df: DataFrame con las variables categóricas transformadas a numéricas
    '''
    
    #Columna Genero
    temp_dict_gender={'Female':1, 'Male':0}
    df['gender']=df.gender.replace(temp_dict_gender)
    
    
    #Columna ever_married
    temp_ever_married={'Yes':1, 'No':0}
    df['ever_married']=df.ever_married.replace(temp_ever_married)
    
    #Columna work_type
    temp_dict_worktype={'children':0, 'Never_worked':1,'Self-employed':3,'Private':4, 'Govt_job':2}
    df['work_type']=df.work_type.replace(temp_dict_worktype)
    
    #Columna Residence_type
    temp_dict_resident={'Urban':1, 'Rural':0}
    df['Residence_type']=df.Residence_type.replace(temp_dict_resident)

    
    #Columna smoking_status
    temp_dict_smoke={'Unknown':0, 'never smoked':1,'formerly smoked':2,'smokes':3}
    df['smoking_status']=df.smoking_status.replace(temp_dict_smoke)

   
    return df


#-------------------------------------------------------------------------------

def guardar_dataset(df, dir, nombre):

    '''
    Esta función guarda un DataFrame en formato csv en la ruta indicada

    Parámetros:
    - df: DataFrame con los datos en quiero guardar en csv
    - dir: ruta donde quiero guardar el archivo
    - nombre: nombre con el que quiero guardar el archivo 
    '''
    
    df.to_csv(os.path.join(dir,nombre), sep=',', index=False)

#-------------------------------------------------------------------------------

def dividir_train_test(df, target):
    '''
    Esta función divide el dataset en X (features) e y (target) y hace el split en datos de entremaniento y test

    Parámetros:
    - df: DataFrame con los datos
    - target: el nombre de la columna target

    Devuelve:
    -X: Dataframe con las features
    -y: Dataframe con el target
    -X_train: DataFrame con las features que vamos a utilizar para entrenar
    -X_test: DataFrame con las features que vamos a utilizar hacer el test
    -y_train: array con el target de entreamiento
    -y_test: array con el target de test

    '''
    X=df.drop(columns=[target])
    y=df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test

#-------------------------------------------------------------------------------
def imputar_media(X_train,df, columna):
    '''
    Esta función reemplaza los missings que hay en una columna con la media del conjunto de train

    Parámetros:
    - X_train: DataFrame con la columna con la que quiero obtener la media
    - df: DataFrame que contiene los missings que quiero reemplazar
    - columna: columna del DataFrame que contie los missings

    Devuelve:
    - df: DataFrame sin missings
    '''
    media = X_train[columna].mean()
    valor_imputado = round(media)
    es_nulo = df[columna].isna()
    df.loc[es_nulo,columna] = valor_imputado
    
    return df
#-------------------------------------------------------------------------------
def feature_reduction(df, lista_columnas):
    '''
    Esta función elimina las columnas que quiero de un DataFrame

    Parámetros:
    - df: DataFrame del que quiero eliminar las columnas
    - lista_columnas: lista con las columnas que quiero eliminar

    Devuelve:
    - df: DataFrame con las columnas eliminadas
    '''
    df.drop(columns=lista_columnas, inplace=True)

    return df
#-------------------------------------------------------------------------------
def escalar_datos(X_train,df):
    '''
    Esta función escala los datos de las variables del DataFrame que se pasa a la función
    - X_train: DataFrame con los datos de entremamiento
    - df: DataFrame con los datos a escalar

    Devuelve:
    - df: DataFrame con los datos escalados
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    df = scaler.transform(df)

    return df
#-------------------------------------------------------------------------------
def train__GridCV_EasyEnsembleClassifier(modelo, X_train, y_train):
    '''
    Esta función entrena el modelo EasyEnsembleClassifier con GridSearhCV
    Parámetros:
    - modelo: el modelo con el que queremos entrenar el EasyEnsembleClassifier
    - X_train 
    - y_train

    Devuelve:
    -best_easyesemcv: modelo entrenado
    '''
    param_grid_dict = {
    'base_estimator': [modelo],
    'sampling_strategy': ['not minority'],
    'replacement': [True],
    'random_state': [0],
    }

    easyesem = EasyEnsembleClassifier()

    easyesemcv = GridSearchCV(easyesem, param_grid_dict, n_jobs=8, cv=5)
    easyesemcv.fit(X_train, y_train)

    best_easyesemcv=easyesemcv.best_estimator_
    best_easyesemcv.fit(X_train, y_train)

    return best_easyesemcv

#-------------------------------------------------------------------------------

def guardar_modelo(nombre, modelo):
    '''
    Esta función guarda el modelo que queremos

    Parámetros: 
    - nombre: nombre con el que se guarda el modelo
    - modelo: modelo a guardar
    '''
    modelo_nombre=nombre
    with open(modelo_nombre, "wb") as archivo_salida:
        pickle.dump(modelo, archivo_salida)

#-------------------------------------------------------------------------------
if __name__ == '__main__':

    print('Este fichero contiene las funciones del proyecto de ML sobre el riesgo de sufrir un accidente cerebro-cardiovascular.')
    sys.exit(0)

#-------------------------------------------------------------------------------
