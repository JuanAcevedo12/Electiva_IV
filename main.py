#Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import statsmodels.formula.api as smf
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import OneRClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

#Importar o cargar Dataset
datos = pd.read_csv("./data_Clasificacion.csv", delimiter=";", engine='python')
pd.set_option("display.max_columns", None)
print("TAMAÑO DATASET: ", )

#Imprimir informacion sobre el DataSet cargado
print(datos.describe())
print(datos.info())
data_clean = datos.copy()
data_clean.columns = data_clean.columns.str.upper()

#Valores faltantes
print("\nVariables con valores nulos\n",data_clean.isnull().sum()[data_clean.isnull().sum() > 0], "\n")

data_clean['NUMERO_HIJOS'] = data_clean['NUMERO_HIJOS'].fillna(round(data_clean['NUMERO_HIJOS'].mean()))
data_clean['KILOMETRAJE_INSTALACION'] = data_clean['KILOMETRAJE_INSTALACION'].replace('-', np.nan)
data_clean['COLOR'] = data_clean['COLOR'].fillna(data_clean['COLOR'].mode()[0])
data_clean['CILINDRAJE'] = data_clean['CILINDRAJE'].fillna(round(data_clean['CILINDRAJE'].mean(), 2))
data_clean['MARCA'] = data_clean['MARCA'].fillna(data_clean['MARCA'].mode()[0])
data_clean['CLASE'] = data_clean['CLASE'].fillna(data_clean['CLASE'].mode()[0])
data_clean['TIPO'] = data_clean['TIPO'].str.replace(r'\[.*\]','', regex=True)
data_clean['TIPO'] = data_clean['TIPO'].str.strip()
data_clean['TIPO'] = data_clean['TIPO'].fillna(data_clean['TIPO'].mode()[0])
data_clean['MODELO'] = data_clean['MODELO'].fillna(data_clean['MODELO'].mode()[0])

#Transformar tipos de datos

data_clean['ID'] = data_clean['ID'].astype('str')
data_clean['EDAD'] = data_clean['EDAD'].round().astype('int')
data_clean['NUMERO_HIJOS'] = data_clean['NUMERO_HIJOS'].astype('int')
data_clean['NUMERO_HIJOS'].unique()
data_clean['KILOMETRAJE_INSTALACION'] = data_clean['KILOMETRAJE_INSTALACION'].str.replace(',', '.').astype('float')
data_clean['KILOMETRAJE_INSTALACION'] = data_clean['KILOMETRAJE_INSTALACION'].fillna(data_clean['KILOMETRAJE_INSTALACION'].mean())
data_clean['MODELO'] = data_clean['MODELO'].astype('int').astype('str')

print("Valores nulos despues:\n", data_clean.isnull().sum()[data_clean.isnull().sum() > 0])

#Convertir valores de las columnas SINIESTROS
data_clean['SINIESTROS_2016'] = data_clean['SINIESTROS_2016'].replace(2, 1)
data_clean['SINIESTROS_2017'] = data_clean['SINIESTROS_2017'].replace(2, 1)
data_clean['SINIESTROS_2018'] = data_clean['SINIESTROS_2018'].replace(2, 1)

#Convertir columnas SINIESTROS, GENERO y SERVICIO a categoricas
data_clean['SINIESTROS_2016'] = data_clean['SINIESTROS_2016'].astype('category')
data_clean['SINIESTROS_2017'] = data_clean['SINIESTROS_2017'].astype('category')
data_clean['SINIESTROS_2018'] = data_clean['SINIESTROS_2018'].astype('category')
data_clean['GENERO'] = data_clean['GENERO'].astype('category')
data_clean['SERVICIO'] = data_clean['SERVICIO'].astype('category')
data_clean['SERVICIO'].unique()

#print(data_clean.info())

# Poner la primera letra MATUSCULA de cada registro para todos los datos no numericos y eliminar espacios
data_clean = data_clean.applymap(lambda x: x.title() if type(x) == str else x)
data_clean = data_clean.applymap(lambda x: x.strip() if type(x) == str else x)

# Ver si hay valores duplicados en las columnas que nos interesan
print(data_clean[data_clean.duplicated(subset=['EDAD', 'EDAD LICENCIA', 'SINIESTROS_2016', 'SINIESTROS_2017', 'SINIESTROS_2018', 
                                         'GENERO', 'SERVICIO', 'MARCA', 'CLASE', 'TIPO', 'MODELO'], keep=False)])


#Creamos nueva columna para evaluar los siniestros y definir si se renueva o no (valores predeterminados)
data_clean = data_clean.assign(RENOVAR_LICENCIA=1)

print(data_clean.info())

#Evaluamos que no tenga ningun siniestro registrado en los ultimos 3 años
data_clean['RENOVAR_LICENCIA'] = data_clean.apply(
    lambda row: row['RENOVAR_LICENCIA'] == 0 
    if row['SINIESTROS_2016'] == 0 and row['SINIESTROS_2017'] == 0 and row['SINIESTROS_2018'] == 0 else row['RENOVAR_LICENCIA'], axis = 1
).astype('int')

# Exportar el dataframe a un archivo csv
data_clean.to_csv('./data_clasificacion_clean.csv', index=False)

#MOSTRAR GRAFICAS PARA ANALISIS PRINCIPAL DE DATASET


# Grafica de barras que muestra por genero cuantas colisiones MAX tuvo
plot.bar(data_clean['GENERO'], data_clean['MAX(COLISION)'])
plot.show()

sns.scatterplot(x=data_clean['MAX(EXCESO VELOCIDAD (MIN))'], y=data_clean['MAX(MAXIMA VELOCIDAD (KM/H))'], data=data_clean)

plot.bar(data_clean['EDAD'], data_clean['MAX(COLISION)'])
plot.show()

#Grafica de barras que muestra para la cantidad de colisiones cuantas frenadas hubo
plot.bar(data_clean['MAX(COLISION)'],data_clean['MAX(FRENADAS (UNI))'])
plot.show()

plot.bar(data_clean['NIVEL_ESTUDIO'], data_clean['MAX(MAXIMA VELOCIDAD (KM/H))'])
plot.show()


#----------------------------------------------------------------
#CLASIFICACION

#Solo columnas de interes para el modelo
data_classificacion = data_clean[['RENOVAR_LICENCIA', 'SINIESTROS_2016', 'SINIESTROS_2017', 'SINIESTROS_2018']]

print(data_classificacion['RENOVAR_LICENCIA'])

# Se dividen los datos en variables independientes X, y variable dependiente y (Renovado)
X = data_classificacion.drop(['RENOVAR_LICENCIA'], axis=1).values
Y = data_classificacion.RENOVAR_LICENCIA.values

#Division de datos - Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# Crear el clasificador
clasif = LogisticRegression()

# Entrenar el clasificador
clasif.fit(X_train, y_train)

# Predecir con el clasificador
y_pred = clasif.predict(X_test)
matriz = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_mat=matriz, figsize=(5, 5), show_normed=True)
plot.xticks([0, 1], ['Si', 'No'])
plot.yticks([0, 1], ['Si', 'No'])
plot.tight_layout()
plot.show()

#Mostrar caracterisitcas del modelo
print(classification_report(y_test, y_pred))

#Arbol de decision
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

arbol = DecisionTreeClassifier()

#Entrenamiento
arbol.fit(X_train, y_train)

#Predecir
y_pred = arbol.predict(X_test)
mat = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_mat=mat, figsize=(5, 5), show_normed=True)
plot.xticks([0, 1], ['Si', 'No'])
plot.yticks([0, 1], ['Si', 'No'])

plot.tight_layout()
plot.show()

# Mostrar características del modelo
print(classification_report(y_test, y_pred))

#ONE R
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# Crear el clasificador
clf = OneRClassifier()

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predecir con el clasificador
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_mat=mat, figsize=(5, 5), show_normed=True)
plot.xticks([0, 1], ['Si', 'No'])
plot.yticks([0, 1], ['Si', 'No'])
plot.tight_layout()
plot.show()

# Mostrar características del modelo
print(classification_report(y_test, y_pred))

#NAIVE BAYES
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# Crear el clasificador
clf = GaussianNB()
# Entrenar el clasificador
clf.fit(X_train, y_train)
# Predecir con el clasificador
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, figsize=(5, 5), show_normed=True)
plot.xticks([0, 1], ['No', 'Si'])
plot.yticks([0, 1], ['No', 'Si'])
plot.tight_layout()
plot.show()
# Mostrar características del modelo
print(classification_report(y_test, y_pred))

#CLUSTERING K MEANS

# Curva elbow para determinar valor óptimo de k.
nc = range(1, 30) # El número de iteraciones que queremos hacer.
kmeans = [KMeans(n_clusters=i) for i in nc]
score = [kmeans[i].fit(data_classificacion).score(data_classificacion) for i in range(len(kmeans))]
score
plot.xlabel('Número de clústeres (k)')
plot.ylabel('Suma de los errores cuadráticos')
plot.plot(nc,score)
plot.show()

kmeans = KMeans(n_clusters=5).fit(data_classificacion)
centroids = kmeans.cluster_centers_
print(centroids)

labels = kmeans.predict(data_classificacion)
data_classificacion['label'] = labels

# Plot k-means clustering.
colores=['red','green','blue','yellow','fuchsia']
asignar=[]
for row in labels:
     asignar.append(colores[row])
plot.scatter(data_classificacion['SINIESTROS_2016'].values+data_classificacion['SINIESTROS_2017'].values+data_classificacion['SINIESTROS_2018'].values, Y, c=asignar, s=1)
plot.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=20) # Marco centroides.
plot.xlabel('Close price')
plot.ylabel('Volume')
plot.title('Samsung stocks k-means clustering')
plot.show()

sns.scatterplot(x=data_classificacion['SINIESTROS_2016'].values, y=Y, data=data_classificacion)

#FORMULAR Y TENER PRESENTE LA PREGUNTA DE CLASIFICACION Y DE ASOSIACION

#Maquina soporte vectorial

#Arbol bosques randomicos
X = data_classificacion.drop(['RENOVAR_LICENCIA'], axis=1).values
Y = data_classificacion.RENOVAR_LICENCIA.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

randomForest = RandomForestClassifier()

randomForest.fit(X_train, y_train)

randomForest.score(X_test, y_test)

y_pred = randomForest.predict(X_test)
matriz = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(conf_mat=matriz, figsize=(6,6), show_normed=False)
plot.tight_layout()
plot.show()

# Mostrar características del modelo
print(classification_report(y_test, y_pred))