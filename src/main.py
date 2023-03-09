#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long

#-------------------------------------------------------------------------------

'''
Este fichero contiene la línea principal del proyecto de ML sobre la predección del riesgo de sufrir un accidente cerebro-cardiovascular.
https://github.com/mariaml92
'''

#-------------------------------------------------------------------------------
#Librerías
import sys
from catboost import CatBoostClassifier
sys.path.append('./utils')
from funciones import cargar_datos, procesar_datos,train__GridCV_EasyEnsembleClassifier, guardar_modelo
from sklearn.metrics import classification_report

#-------------------------------------------------------------------------------



# Cargar los datos brutos
enlace= './data/raw_data/healthcare-dataset-stroke-data.csv'
df = cargar_datos(enlace)


# Procesar datos
df, X, y, X_train_scaled, X_test_scaled, y_train, y_test = procesar_datos(df)


# Entrenar modelo
best_easyesemcv = train__GridCV_EasyEnsembleClassifier(CatBoostClassifier(verbose=False, n_estimators=30), X_train_scaled, y_train)


# Guardar el modelo
guardar_modelo('EasyEnsembleCatboostcv.model', best_easyesemcv)









   


#-------------------------------------------------------------------------------

