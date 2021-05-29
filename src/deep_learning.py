
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
#from tensorflow.keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD



#
# Lectura y visualización del set de datos
#

datos = pd.read_csv('datasetseg.csv', sep=",", skiprows=0, usecols=[4,5])
print(datos)

# Al graficar los datos se observa una tendencia lineal
datos.plot.scatter(x='poblacion', y='total')
#plt.scatter(datos['total'].values,y)
plt.xlabel('Poblacion ')
plt.ylabel('Contagiados')
plt.show()

x = datos['poblacion'].values
y = datos['total'].values

#
# Construir el modelo en Keras
#

# - Capa de entrada: 1 dato (cada dato "x" correspondiente a la edad)
# - Capa de salida: 1 dato (cada dato "y" correspondiente a la regresión lineal)
# - Activación: 'linear' (pues se está implementando la regresión lineal)

np.random.seed(2)			# Para reproducibilidad del entrenamiento

input_dim = 1
output_dim = 1
modelo = Sequential()
#modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

#INI OPAC
modelo.add(Dense(1000, input_dim=1, activation='linear'))
modelo.add(Dense(100, activation='linear'))
modelo.add(Dense(10, activation='linear'))
modelo.add(Dense(1, activation='linear'))
#modelo.add(Dense(10, activation='elu'))
#FIN OPAC

# Definición del método de optimización (gradiente descendiente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático
# medio40000

#sgd = SGD(lr=0.0004)
#modelo.compile(loss='mse', optimizer=sgd)


#INI OPAC
modelo.compile(loss='mean_squared_error',
        optimizer='adam',
        metrics= ['binary_accuracy'])
#FIN OPAC

# Imprimir en pantalla la información del modelo
modelo.summary()

#
# Entrenamiento: realizar la regresión lineal
#

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
# iteración (batch_size = 29)

num_epochs = 10000
batch_size = x.shape[0]
#history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

#INI OPAC
history = modelo.fit(x, y, epochs=num_epochs, verbose=0)
#FIN OPAC
#
# Visualizar resultados del entrenamiento
#

# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Graficar el error vs epochs y el resultado de la regresión
# superpuesto a los datos originales
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
#plt.title('ECM vs. epochs')

y_regr = modelo.predict(x)
plt.subplot(1,2,2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('Poblacion')
plt.ylabel('Contagiados')
#plt.title('Datos originales y regresión lineal')

# Al graficar los datos se observa una tendencia lineal
#plt.subplot(1,3,3)
#datos.plot.scatter(x='poblacion', y='total')
#plt.scatter(y,x)
#plt.xlabel('Poblacion ')
#plt.ylabel('total')

plt.show()

print("Batch Size", batch_size)

# Predicción
x_pred = np.array([100083])
y_pred = modelo.predict(x_pred)
print("La cantidad de {:.1f}".format(y_pred[0][0]), " personas contagiadas de Covid habran para una poblacion de {} personas".format(x_pred[0]))