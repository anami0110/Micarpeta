# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:08:09 2023

@author: anami
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from time import time 
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

# Definir matriz de confusion
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# Probar la red
accuracies = []
confmatrix = []
pred = []
true =[]
test_data = np.load('DSfold9_comprimido.npz')
x_test = test_data["features"]
y_test = test_data["labels"]

# Cargar modelo
interpreter = tf.lite.Interpreter(model_path="nq_fold9.tflite")
interpreter.allocate_tensors()

# Test
y_true, y_pred = [], []
total = x_test.shape[0]

for x, y in zip(x_test, y_test):
    y = y.reshape(1)
    y = int(y)
    x = x.reshape(1,41,79,2)

    # Predicción de la ventana
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    avg_p = np.argmax(interpreter.get_tensor(output_details[0]['index']))

stop = time()
true.append(y_true)
pred.append(y_pred)
accuracies.append(accuracy_score(y_true, y_pred))   
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

"""import numpy as np
from tflite_runtime.interpreter import Interpreter

# Carga el modelo TFLite
interpreter = Interpreter(model_path="nq_fold9.tflite")
interpreter.allocate_tensors()

# Obtiene los detalles del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carga los datos de entrada desde el archivo npz
with np.load('DSfold.npz') as data:
    input_data = data['arr_0']

# Ejecuta la inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Imprime los resultados
print(output_data) """

