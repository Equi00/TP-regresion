# En este trabajo practico se analizara un dataset que indica informacion importante sobre una gran cantidad de animes.
# Las tablas que tenemos en nuestro dataset son el nombre, genero, tipo, episodios, rating y miembros.
# Como objetivo queremos poder predecir el rating de los animes con los datos del genero, tipo, episodios y miembros.
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/foyol/Desktop/paradigmas/TP-regression/dataset/anime.csv")

print("\nDESCRIPCION DE LA TABLA")
print(df.describe())

print("\nCOLUMNAS")
print(df.columns)

print("\nPRIMEROS DATOS DE LA TABLA")
print(df.head())

print("\nINFORMACION DE LA TABLA")
print(df.info())

# Observamos que en la tabla hay columnas con distinta cantidad de filas, esto quiere decir que tenemos valores null
# en la tabla de datos.

print("\nCANTIDAD DE VALORES NULL POR COLUMNA")
print(df.isnull().sum())

# Ahora observamos mejor la cantidad de valores nulos por columna,
# de los cuales 2 son columnas categoricas y 1 es numerica. Realizaremos un grafico para poder observarlo mejor.

sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Se observa que la mayoria de valores null de 'rating' estan concentrados en entre los datos 10800 y 11400
# aproximadamente, tambien la mayoria de los valores null de las columnas 'genre' y 'type' estan tambien dentro de
# ese rango.

df_rating_null = df[df['rating'].isnull()]
df_genre_null = df[df['genre'].isnull()]
df_type_null = df[df['type'].isnull()]

print("\nDATOS DE LA TABLA CON VALORES RATING NULL")
print(df_rating_null[['name', 'members']].head())
print("\nDATOS DE LA TABLA CON VALORES GENRE NULL")
print(df_genre_null[['name', 'members']].head())
print("\nDATOS DE LA TABLA CON VALORES TYPE NULL")
print(df_type_null[['name', 'members']].head())


# Por lo visto en los valores iniciales de los datos cuyas columnas genre, type y rating son null se observa que
# estos animes no son desconocidos o muy viejos. Tenemos por ejemplo a 'Steins;Gate 0', 'Code Geass: Fukkatsu no
# Lelouch', 'Violet Evergarden', 'Free! (Shinsaku)', 'IS: Infinite Stratos 2 - Infinite Wedding', 'One Punch Man 2' y
# 'Gintama (2017)'. Estos animes son muy populares tanto dentro de Japon como de forma internacional. Podemos
# entonces decir que el hecho de que estos registros tengan variables null se debe mas a un error que a razones como:
# el anime es muy viejo, el anime no fue visto por nadie o por muy poca gente, etc. Podria ser tambien
# que al ingresar estos animes en la base de datos se hayan olvidado de poner ciertos datos importantes,
# ya sea porque los encargados no supieran de que tipo o genero es el anime.

# Como la cantidad de datos nulos en comparacion con la cantidad de datos totales de la tabla es muy pequenia,
# procederemos a eliminar estas filas con valores nulos.


def drop_na(data):
    data = data.dropna(axis=0)

    print("\nCANTIDAD DE VALORES NULL POR COLUMNA")
    print(data.isnull().sum())

    sns.heatmap(data.isnull(), cbar=False)
    plt.show()

    return data


df = drop_na(df)


# Ahora que no tenemos mas valores nulos, procederemos a analizar los datos de las columnas,

# Se visualiza que la columna 'episodes' es una variable categorica, pero deberia ser una variable numerica. Procederemos
# a cambiar el tipo de dato a entero.

def to_int(val):
    """ Reconoce valores numericos y los transforma a enteros.
    """
    try:
        value = int(float(val))
    except ValueError:
        value = 0
    return value


values = df['episodes'].values
df['episodes'] = pd.DataFrame(values, columns=["episodes"])

df['episodes'] = df['episodes'].map(to_int)

print("\nVALORES NUMERICOS DE EPISODIOS")
print(df['episodes'])

# Logramos convertir todos los valores de la tabla 'episodes' en numeros, pero observamos que en la tabla habian valores
# que eran strings vacios. Al convertir todos los strings a numeros, los vacios se convirtieron en ceros. Estos datos
# en el dataset pueden deberse a que la persona a cargo de ingresar los datos olvido poner la cantidad de capitulos que
# tiene el anime, o pudo ingresar el anime al dataset cuando aun estaba en emision, por lo que la persona no sabria que
# cantidad de capitulos podria tener el anime en su momento.
# En todo caso, vamos a visualizar la cantidad de valores cero que tenemos.

print("\nCANTIDAD DE CEROS EN LA COLUMNA episodes")
zero_values = df[df == 0].count(axis=0)
print(zero_values[zero_values > 0])

# Pasamos los valores cero a null para asi poder graficarlos.

df = df.replace(0, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Observamos una cantidad de 433 valores faltantes en la columna de episodes, estos se encuentran mayoritariamente
# entre los valores 11800 hasta los 12000 aproximadamente. Como la cantidad de datos null en episodios es un 3.6% de
# la cantidad total de datos en la tabla, procederemos a eliminar estos datos null.

df = drop_na(df)

# Ahora que tenemos todos nuestros datos en orden, seguimos con el analisis de datos.

# Procedemos a ver la informacion que nos brinda nuestras columnas categoricas

print("\nDESCRIPCION DE LAS COLUMNAS name, genre y type")
print(df[['name', 'genre', 'type']].describe())

# Se observa que hay un total de 11582 nombres unicos, tenemos un total de 3182 generos/ combinacion de
# generos, una cantidad de 6 tipos. Tambien vemos que la mayoria de anime son programas de TV, osea las series.
# El genero con mas frecuencia es el Hentai, con una frecuencia del 630, pero si tenemos en cuenta que los animes tienen
# multiples generos, entonces este resultado puede estar mal.

# Ahora mostraremos los 5 animes con mayor rating, los 5 animes con mayor cantidad de miembros y los 5 animes con mayor
# cantidad de episodios, luego sacaremos conclusiones de lo visto.

df_mejores5 = df[['name', 'rating', 'members', 'episodes']]

print("\n5 ANIMES CON MAYOR RATING")
print(df_mejores5.sort_values(by="rating", ascending=False).head())

print("\n5 ANIMES CON MAYOR CANTIDAD DE MIEMBROS")
print(df_mejores5.sort_values(by="members", ascending=False).head())

print("\n5 ANIMES CON MAYOR CANTIDAD DE EPISODIOS")
print(df_mejores5.sort_values(by="episodes", ascending=False).head())

# De la tabla de animes con mayor rating, se puede observar que estos animes, a excepcion de 'Kimi no Na wa',
# tienen una cantidad de miembros muy pequenia, esto significa que muchos de los animes mejores valorados no son muy
# conocidos pero para los que lo vieron les parecio excelente. Seguramente si hubieran muchisimos mas miembros en
# estos animes el rating cambiaria por las distintas opiniones y gustos de las personas.

# En la tabla con la mayor cantidad de miembros podemos ver los animes mas populares tanto dentro de japon como
# internacionalmente. Tambien observamos que a mayor cantidad de miembros, mas variado es el rating de estas series.

# En la tabla con la mayor cantidad de capitulos podemos observar que la cantidad de miembros de los mismos no es muy
# alta, esto se puede deber a que no mucha gente esta dispuesta a ver series extremadamente largas. Tambien vemos que la
# puntuacion que tienen estos animes en el rating es bastante promedio, por lo que tener una gran cantidad de capitulos
# pareciera ser mas perjudicial que bueno, ya que estas series sufren de la repetitividad y falta de imaginacion de los
# autores, generando un desgaste en las personas que ven estas series.

# Antes de avanzar con el analisis vamos a descartar la columna con el ID de los animes, ya que no nos sirve de nada
# tener ese dato.

df = df[['name', 'genre', 'type', 'episodes', 'rating', 'members']]
print("\nDATASET SIN LA COLUMNA DE ID")
print(df.head())

# Procedemos a ver la informacion que nos brinda nuestras columnas numericas

print("\nDESCRIPCION DE LAS COLUMNAS episodes, rating y members")
print(df[['episodes', 'rating', 'members']].describe())

# Por parte de los episodios vemos que el promedio esta entre 12 y 13 capitulos, esto nos puede decir que la mayoria
# de los animes tienen esta cantidad de capitulos, lo cual es una cantidad comercial mas comoda para las empresas
# animadoras, puesto a que muchos animes funcionan mas que nada para hacer una publicidad al manga(comic japones).
# Ademas, se gasta menos presupuesto haciendo menos capitulos. En la industria muy pocos animes llegan a tener
# temporadas de muchos capitulos.

# Por parte del rating vemos que el promedio esta entre 6 y 7, esto nos dice que segun la gente que ve anime,
# la mayoria de estos son promedio o entretenidos, son pocos los animes malos o excelentes. Esto se puede deber
# tambien a que en la industria del anime no siempre se quiere hacer el mejor producto, como dijimos antes el anime
# mayoritariamente es un producto comercial.

# Por parte de la cantidad de miembros, que hablan sobre un anime en especifico, vemos que el promedio esta cerca de
# los 19000, es una cantidad bastante considerable de gente la que mira anime. cabe a destacar que el valor 'std' en
# las columnas de miembros y episodios es bastante alto, lo que indica una varianza alta. Reafirmaremos todas estas
# conclusiones observando el grafico de distribucion de datos.

# Gráfico de distribución para cada variable numérica
# ==============================================================================

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes = axes.flat

columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data=df,
        x=colum,
        stat="count",
        kde=True,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        line_kws={'linewidth': 2},
        alpha=0.3,
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=8)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución de variables numéricas', fontsize=10, fontweight="bold")
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(columnas_numeric):
    axs[i].hist(x=df[colum], bins=20, color="#3182bd", alpha=0.5)
    axs[i].plot(df[colum], np.full_like(df[colum], -0.01), '|k', markeredgewidth=1)
    axs[i].set_title(f'Distribución {colum}')
    axs[i].set_xlabel(colum)
    axs[i].set_ylabel('counts')

plt.tight_layout()
plt.show()

# Como podemos observar, nuestras conclusiones sobre los datos eran correctas. Tenemos que la mayoria de animes tienen
# pocos capitulos, la mayoria de animes no tienen una cantidad de miembros grande y el rating promedio de la mayoria
# de animes esta entre los valores 6 y 7.

# Análisis Univariado

# Medidas de centralización: media, mediana y moda

for i, colum in enumerate(columnas_numeric):
    print(f"\nMEDIDAS DE CENTRALIZACION {colum}")
    print(f'Media:{df[colum].mean()} \
     \nMediana: {df[colum].median()} \
     \nModa: {df[colum].mode()}')

# Vemos que en todos los casos MEDIA > MEDIANA > MODA, esto significa que hay una distribucion sesgada positivamente
# de los datos de todas las variables numericas

# Medidas de dispersión: desviación típica, rango, IQR, coeficiente de variación, desviación media

print(f'\nLa varianza es:\n{df.var()}')

# La varianza de rating es muy pequenia, por lo que no tiene mucha dispersion de los datos. En cambio los datos de
# episodios y miembros tenemos una varianza alta, lo que indica una gran dispersion de los datos.

print(f'\nDesviación Estándar por fila:\n{df.std(axis=0)}')

for i, colum in enumerate(columnas_numeric):
    print(f"\nRANGO DE {colum}")
    print(f'El rango es: {df[colum].max() - df[colum].min()}')

for i, colum in enumerate(columnas_numeric):
    print(f"\nEL RANGO INTERCUATRILICO DE {colum}")
    print(f'El IQR es: {df[colum].quantile(0.75) - df[colum].quantile(0.25)}')

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
df_varianza = df[['rating', 'episodes', 'members']].apply(cv)
print(
    f'\nEl coeficiente de variación es:\n{df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"]).apply(cv)}')

# Mediante la desviacion estandar, el rango, el rango intercuatrilico y el coeficiente de variacion tambien se observa
# que hay una alta dispersion en episodios y miembros.

# Medidas de asimetría

print(f"\nLas medidas de asimetría son:\n{df.skew()}")

# la medida de asimetria nos muestra que episodios tiene mas valores extremos positivos que negativos, lo mismo pasa
# con la columna members.

print(f"\nLas medidas de kurtosis son:\n{df.kurt()}")

# Segun las medidas de kurtosis, las columnas de episodios y miembros son de distribucion leptocurtica. La columna
# rating tiene una distribucion platicurtica

# Ya hemos visto y analizado la distribucion de los datos numericos, lo proximo que vamos a analizar es la cantidad de
# anime que hay de cada genero y de cada tipo y sacar conclusiones. Pero tenemos un problema, los animes pueden ser de
# multiples generos, por lo que primero debemos separar cada genero y contarlos por separado.

# separamos los generos y los ponemos dentro de una lista
all_genres = []
for item in df['genre']:
    item = item.strip()
    all_genres.extend(item.split(', '))

# Contamos el numero de items que hay en la lista
c = Counter(all_genres)

# Realizamos el grafico de cada genero con su cantidad
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.countplot(all_genres)

plt.title('Cantidad de series por género')
plt.xlabel('Género')
plt.ylabel('Cantidad de series')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# En este grafico se puede observar que la mayoria de los animes del dataset tienen la comedia como uno de sus
# generos, esto nos dice que la industria del anime ingresa en cada serie algo de comedia. El resto de generos con
# una gran frecuencia son: accion, aventura, fantasia, ciencia ficcion y drama. Todo esto tiene sentido,
# ya que el anime tiene como publico objetivo a las personas de entre 13 y 17 anios. la gente en esa edad esta mas
# interesada en cosas que sean entretenidas e interesantes mayoritariamente.

# Una vez analizada la cantidad de series por genero, haremos unos graficos para comparar y ver como influyen nuestras
# variables numericas con el tipo de anime, ya sea serie, pelicula, ova, etc.

for column in columnas_numeric:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    sns.barplot(x='type', y=column, data=df, ax=ax)

    plt.title(f'{column} segun el tipo de anime')
    plt.xlabel('Tipo')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

# En el grafico de episodios segun el tipo de anime, se ve que las series de television dominan en cantidad de
# capitulos, el resto de tipos tienen una cantidad bastante nivelada entre si, la media no llega a los 10 capitulos,
# tambien podemos ver que las peliculas tienen una media de 9 capitulos aproximadamente, esto puede deberse a que hay
# peliculas de anime, las cuales las convierten en un formato de serie para volver a lanzarla.

# En el grafico de rating segun el tipo de anime, en todos los tipos el rating promedio esta bastante nivelado,
# dominando las series anime de television por muy poco. el valor promedio de rating de cada tipo de anime ronda
# entre 5 y 7.

# En el grafico de miembros segun el tipo de anime, vemos una dominancia absoluta en el tipo de series de television.
# Tiene sentido, ya que las peliculas no suelen ser muy populares y son momentos fugases para los espectadores,
# en cambio las series de anime estan en emision durante meses y pueden conseguir a muchos seguidores y gente que se
# interese por el boca a boca de los espectadores, las discusiones sobre la serie, los memes que sacan de la misma y
# las ganas de ver el siguiente capitulo. Tambien la duracion de los capitulos, 20 minutos aproximadamente,
# hacen que los expectadores no se abrumen con la duracion pero tampoco van a quedar totalmente satisfechos. La
# implementacion de esta duracion de los capitulos hace que en cada uno de estos haya un buen ritmo dependiendo el
# genero de la misma serie. En occidente estamos acostumbrados a que las series duren 1 hora aproximadamente,
# al ser capitulos tan prolongados el ritmo del mismo puede cambiar, incluso puede ser agobiante estar tanto tiempo
# mirando una serie. El formato que utilia Japon es perfecto para mirar un capitulo en un recreo o mientras estas
# almorzando sin la necesidad de pausar en medio del capitulo por falta de tiempo.

# Ahora vamos a comprobar estos datos de una forma mas exacta para ver si los graficos estan bien.

print("\nVEMOS SI EL GRUPO ES EQUILIBRADO")
print(df.groupby('type').size())

# Se puede observar que las agrupaciones por tipo no son equilibradas, hay una gran cantidad de peliculas, ovas y series
# Dejando a los anime de tipo musical y ONA  como minoritarios.

print("\nMEDIA Y DESVIACION TIPICA POR GRUPO\n")
for column in columnas_numeric:
    print(f"\nMEDIA Y DESVIACION TIPICA DE {column}")
    print(df.groupby('type')[column].agg(['mean', 'std']).round(2))

# Tenemos que hay desviacion tipica en los episodios de cada tipo, hay muy poca desviacion tipica de los rating de cada
# tipo y tenemos una gran desviacion tipica por partes de los miembros por tipo. Tambien vemos que los datos de la media
# es la misma que figura en los graficos.

# Como la cantidad de tipos de anime es mucho mas pequenia que la cantidad de generos, se puede hacer el siguiente
# grafico para poder visualizar, junto con los porcentajes la cantida de animes por tipo.

fig, ax = plt.subplots(1, 1, figsize=(15, 8))
type_data = Counter(df.type)
labels = list(type_data.keys())
sizes = list(type_data.values())

ax.pie(sizes, labels=labels, shadow=False, startangle=0, autopct="%1.2f%%")
ax.axis('equal')
ax.set_title("Grafico por tipo de anime")
plt.show()

# Antes de realizar alguna regresion, debemos observar que nuestras variables numericas no tengan outliers.

def plot_boxplot(df, ft):
    sns.boxplot(y=ft, data=df)
    plt.show()



for column in columnas_numeric:
    plot_boxplot(df, column)

# Bueno, hay una gran cantidad de outliers en las columnas de members y episodes, en rating tambien hay outliers pero
# no son tantos. Como la cantidad de outliers parece ser muy grande, eliminar todos estos datos puede ser malo, por lo
# que guardaremos el dataset previo a su eliminacion.

df_normal = df


def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr

    ls = df.index[(df[ft] < low) | (df[ft] > up)]

    return ls


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


def delete_outliers(df, list):
    n = 100
    df_clean = df
    for i in range(n):
        index_list = []
        for feature in list:
            index_list.extend(outliers(df_clean, feature))
        if not index_list:
            break
        df_clean = remove(df_clean, index_list)
    return df_clean


df_sin_outlier = delete_outliers(df, columnas_numeric)

print("\nTABLA SIN LOS OUTLIERS")
print(df_sin_outlier.shape)
print(df_sin_outlier.info())

for column in columnas_numeric:
    plot_boxplot(df_sin_outlier, column)

# La eliminacion de outliers fue exitosa, sin embargo, nos quedamos con menos de 2000 registros del dataset,
# se eliminaron demaciados datos y esto podria no ser bueno para nuestras predicciones. Como precaucion vamos a tener
# el mismo dataset pero esta vez con los outliers reemplazados por el valor medio de cada tabla numerica. De esta forma
# no perdemos datos. la desventaja es que cambiaremos la distribucion de los datos y esto podria provocar sesgos.


def replace(df, list):
    column_means = df.mean()

    column_std = df.std()

    outlier_threshold = 3

    for column in list:
        upper_limit = column_means[column] + (outlier_threshold * column_std[column])
        lower_limit = column_means[column] - (outlier_threshold * column_std[column])

        df[column] = np.where((df[column] > upper_limit) | (df[column] < lower_limit), column_means[column], df[column])
    return df


def replace_outliers_mean(df, list):
    n = 100
    df_clean = df
    index_list2 = []
    for i in range(n):
        index_list = []
        for feature in list:
            index_list.extend(outliers(df_clean, feature))
        if not index_list:
            break
        if index_list == index_list2:
            break
        index_list2 = index_list
        df_clean = replace(df_clean, list)
    return df_clean


df_outlier_reemplazado = replace_outliers_mean(df, columnas_numeric)

print("\nTABLA CON OUTLIER REEMPLAZADOS")
print(df_outlier_reemplazado.shape)
print(df_outlier_reemplazado.info())

for column in columnas_numeric:
    plot_boxplot(df_outlier_reemplazado, column)

# reemplazamos los outliers de forma exitosa, sin embargo nos aparecen unos pocos outliers, de los cuales no se pudo
# reemplazar, procedemos a eliminarlos.

df_outlier_reemplazado = delete_outliers(df, columnas_numeric)

print("\nTABLA CON OUTLIER REEMPLAZADOS")
print(df_outlier_reemplazado.shape)
print(df_outlier_reemplazado.info())

for column in columnas_numeric:
    plot_boxplot(df_outlier_reemplazado, column)

# Ahora tenemos 3 datasets, una con todos los outliers, otra con los outliers eliminados y otra con los outliers
# reemplazados por el valor medio de la columna.

# a continuacion veremos las correlaciones entre nuestras columnas.

# Heatmap matriz de correlaciones
print("\nMATRIZ DE CORRELACIONES CON OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nMATRIZ DE CORRELACIONES SIN OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_sin_outlier.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nMATRIZ DE CORRELACIONES CON OUTLIERS REEMPLAZADOS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_outlier_reemplazado.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()


# Despues de visualizar las correlaciones vemos que la columna rating tiene una mayor correlacion con la columna miembros

# A continuacion realiaremos las regresiones, empezaremos con la regresion lineal en los 3 dataset que tenemos.
# Primero que nada debemos convertir las variables categoricas a numericas.

def linear_regression(df):
    df_genero_encoded = pd.get_dummies(df['genre'], prefix='genre')
    df_tipo_encoded = pd.get_dummies(df['type'], prefix='type')

    # ingresamos las columnas codificadas
    df_encoded = pd.concat([df, df_genero_encoded], axis=1)
    df_encoded = pd.concat([df_encoded, df_tipo_encoded], axis=1)

    X = df_encoded.drop(['name', 'genre', 'rating', 'type'], axis=1)  # Variables predictoras
    y = df_encoded['rating']  # Variable objetivo

    # generamos nuestros conjuntos de entrenamiento y test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("\nCHEQUEAMOS DIMENSIONES DE CONJUNTO DE ENTRENAMIENTO Y DE TESTING")
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # ajustar el modelo de RLM con el conjunto de entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predicción de los resultados en el conjunto de testing.
    y_pred = model.predict(X_test)

    df_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df_prediction.head(25)
    print(df1.head())

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='darkgreen')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print("\nPREDICCION DE DATASET NORMAL")
linear_regression(df)

# Por parte del dataset normal, vemos que el error es considerablemente alto, por lo que nuestro modelo no puede hacer
# buenas predicciones sobre este dataset sin modificar, esto puede deberse a la cantidad de datos extremos que hay en
# las columnas.

print("\nPREDICCION DE DATASET SIN OUTLIERS")
linear_regression(df_sin_outlier)

# Por parte del dataset sin los outliers, vemos que el error es menor, sin embargo sigue siendo muy grande, por lo que
# nuestro modelo no puede hacer buenas predicciones sobre este dataset sin outliers. Esto puede deberse a que no
# disponemos de los suficientes datos como para hacer una buena prediccion.

print("\nPREDICCION DE DATASET CON OUTLIER REEMPLAZADOS")
linear_regression(df_outlier_reemplazado)

print("\nDESCRIPCION DEL DATASET CON OUTLIER REEMPLAZADOS")
print(df_outlier_reemplazado.describe())

# Por parte del dataset con los outliers reemplazados, vemos una gran mejora en nuestro modelo, el error cuadratico
# medio constituye el 10.4% del valor medio del rating en el dataset con los outliers reemplazados. Esto quiere decir
# que nuestro modelo no es muy preciso, aun asi es capaz de realizar buenas predicciones. Esta franja de error es
# aceptable, pero puede haber otro tipo de regresion que de mejores resultados.

# Vamos a probar con la regresion polinomica, como tenemos varias variables en X, graficaremos con grafico de barras.
# En esta ocacion queremos predecir el rating promedio de los tipos de anime.


def poli_regression(df, number):
    label_encoder = LabelEncoder()

    df_encoded = df

    df_encoded['gennre_encoded'] = label_encoder.fit_transform(df['genre'])
    df_encoded['type_encoded'] = label_encoder.fit_transform(df['type'])

    X = df_encoded[['gennre_encoded', 'type_encoded', 'episodes', 'members']]  # Variables predictoras
    y = df_encoded['rating']  # Variable objetivo

    poly_reg = PolynomialFeatures(degree=number)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    y_pred = lin_reg_2.predict(X_poly)

    df['prediccion_rating'] = y_pred
    rating_promedio = df.groupby('type')['rating'].mean()
    prediccion_promedio = df.groupby('type')['prediccion_rating'].mean()

    tipos = rating_promedio.index
    x = np.arange(len(tipos))
    ancho_barra = 0.35

    fig, ax = plt.subplots()
    barras_rating = ax.bar(x - ancho_barra / 2, rating_promedio, ancho_barra, label='Rating promedio')
    barras_prediccion = ax.bar(x + ancho_barra / 2, prediccion_promedio, ancho_barra, label='Predicción promedio')

    ax.set_xlabel('Tipo')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación del Rating promedio y Predicción promedio por Tipo')
    ax.set_xticks(x)
    ax.set_xticklabels(tipos)
    ax.legend()
    plt.show()

    print(f"\nREGRESION GRADO {number}")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))



poli_regression(df, 2)
poli_regression(df, 3)
poli_regression(df, 4)
poli_regression(df, 5)

# Se puede observar que a partir del grado 5 para arriba, el valor del error cuadratico medio aumenta, por lo que el
# mejor grafico polinomial es el de grado 4. Mediante esta regresion observamos que el error cuadratico medio es del
# 10.7% aproximadamente. Esta prediccion no esta del todo mal teniendo en cuenta que se utilizo el dataset sin
#  modificar.

poli_regression(df_sin_outlier, 2)
poli_regression(df_sin_outlier, 3)
poli_regression(df_sin_outlier, 4)

print("\nDESCRIPCION DEL DATASET SIN OUTLIERS")
print(df_sin_outlier.describe())

# Se puede observar que a partir del grado 4 para arriba el valor del error cuadratico medio aumenta, por lo que el
# mejor grafico polinomial es el de grado 3. Mediante esta regresion observamos que el error cuadratico medio es del
# 13% del dataset sin outliers aproximadamente, el error es mayor en el dataset sin los outliers que en el dataset
# normal. Esto puede deberse a la poca cantidad de datos que tiene el dataset.

poli_regression(df_outlier_reemplazado, 2)
poli_regression(df_outlier_reemplazado, 3)
poli_regression(df_outlier_reemplazado, 4)
poli_regression(df_outlier_reemplazado, 5)
poli_regression(df_outlier_reemplazado, 6)

print("\nDESCRIPCION DEL DATASET CON OUTLIER REEMPLAZADOS")
print(df_outlier_reemplazado.describe())

# Se puede observar que a partir del grado 6 para arriba el valor del error cuadratico medio aumenta, por lo que el
# mejor grafico polinomial es el de grado 5. Mediante esta regresion observamos que el error cuadratico medio es del
# 9.9% del dataset con outliers modificados aproximadamente. el error es mejor que en el resto de datasets. Este valor
# de error es aceptable ya que esta por debajo del 10%, aunque sea por un poco.

# Por ultimo probaremos con los arboles de decision y random forest para poder predecir el rating de los animes.

def show_fit(regressor, title, df):
    label_encoder = LabelEncoder()

    df_encoded = df

    df_encoded['genre_encoded'] = label_encoder.fit_transform(df['genre'])
    df_encoded['type_encoded'] = label_encoder.fit_transform(df['type'])
    X = df_encoded[['genre_encoded', 'type_encoded', 'episodes', 'members']]
    y = df_encoded['rating']

    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    df_prediction = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    df1 = df_prediction.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.xlabel('Índice del anime')
    plt.ylabel('Rating')
    plt.title(f'Predicción del rating utilizando {title}')
    plt.legend()
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

    # Para saber si hay sobreajuste hacemos lo siguiente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor.fit(X_train, y_train)

    cv_scores = cross_val_score(regressor, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    print("Puntuaciones de validación cruzada:", cv_scores)
    print("Puntuación media de validación cruzada:", mean_cv_score)
    test_score = regressor.score(X_test, y_test)
    print("Puntuación en los datos de prueba:", test_score)

    # De esta forma podemos saber si difieren mucho los valores de la puntuacin de validacion cruzada y la puntuacion
    # de los datos de prueba.
    print("DIFERENCIA: ", test_score - mean_cv_score)  # La diferencia indica si hay o no sobreajuste de los datos


arboles_regresion = DecisionTreeRegressor(random_state=0)
arboles_random_regresion = RandomForestRegressor(n_estimators=300, random_state=0)

print("\nPREDICCION CON ARBOLES DE DECISION EN DATASET NORMAL")
show_fit(arboles_regresion, 'Arboles de decision', df)

# Observamos que el arbol de decision nos da un error muy pequenio, osea que la prediccion es muy exacta, sin embargo
# tambien vemos que hay una diferencia significativa entre la puntuacion de validacion cruzada y la puntuacion de los
# datos de prueba, entonce hay sobreajuste. Para sacar este sobre ajuste se va a controlar el nivel de profundidad de
# nuestro arbol de decision.

max_depth = 3
for i in range(7):
    print("\nPROFUNDIDAD ", max_depth)
    arboles_regresion_limitado = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    show_fit(arboles_regresion_limitado, 'Arboles de decision', df)
    max_depth = max_depth + 1

# Como podemos ver, en el nivel de profundidad 9 tenemos una diferencia de puntuacion cruzada y puntuacion de prueba
# aceptable, a mayor profundida habra mayor diferencia, por lo tanto mayor sobre ajuste. En esta profundidad nos
# encontramos con un error del 9.27% con el dataset normal, este valor es bastante bueno y nos dice que nuestro modelo
# puede hacer buenas predicciones.

print("\nPREDICCION CON RANDOM FOREST EN DATASET NORMAL")
show_fit(arboles_random_regresion, 'Random Forest', df)

# Con random forest en el dataset normal podemos observar que no hay sobreajuste, el error es del 4%, lo que nos da un
# modelo con alta precision en sus predicciones a compraracion con la regresion de arboles de decision.

print("\nPREDICCION CON ARBOLES DE DECISION EN DATASET SIN OUTLIERS")
show_fit(arboles_regresion, 'Arboles de decision', df_sin_outlier)

# Podemos observar que la regresion con arboles de decision se dan buenos resultados en las predicciones, dando un error
# muy bajo, sin embargo se puede ver tambien que hay sobreajuste, la diferencia entre la puntuacion de validacion
# cruzada y la puntuacion de los datos de prueba es considerable. Tambien el ver que las puntuaciones media de la
# validacion cruzada es negativa y mucho menor, deja al descubierto que el modelo no generaliza bien a datos no vistos
# previamente. El modelo no puede generalizar nuevos datos.
# Esto puede deberse a la poca cantidad de registros en el dataset con los outliers eliminados, probemos si modificando
# el nivel de profundidad del arbol los resultados pueden mejorar.

max_depth = 3
for i in range(2):
    print("\nPROFUNDIDAD ", max_depth)
    arboles_regresion_limitado = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    show_fit(arboles_regresion_limitado, 'Arboles de decision', df_sin_outlier)
    max_depth = max_depth + 1

# Como podemos ver, en el nivel de profundidad 3 tenemos una diferencia de puntuacion cruzada y puntuacion prueba
# aceptable, por no lo que no hay sobreajuste, a mayor profundidad mayor diferencia. Vemos que el error es del 13%
# aproximadamente en el dataset sin outliers, es un valor aceptable, el modelo puede hacer buenas predicciones.
# Pero la cantidad de datos es muy baja y el arbol solo puede darnos predicciones aceptables a muy poca profundidad.
# Por lo que este modelo no es recomendable.

print("\nPREDICCION CON RANDOM FOREST EN DATASET SIN OUTLIERS")
show_fit(arboles_random_regresion, 'Random Forest', df_sin_outlier)

# Podemos observar que con random forest tenemos buenas predicciones y un valor de error bajo. Sin embargo tenemos
# tambien sobreajuste. Tambien el ver que la puntuacion media de validacion cruzada sea negativa quiere decir que
# nuesto modelo no es capaz de generalizar bien datos no vistos previamente. Esto se debe a que la cantidad de datos
# de nuestro dataset es muy baja. Probemos si modificando el nivel de profundidad podemos mejorar estos resultados.

max_depth = 3
for i in range(4):
    print("\nPROFUNDIDAD ", max_depth)
    arboles_regresion_limitado = RandomForestRegressor(n_estimators = 300, random_state = 0, max_depth=max_depth)
    show_fit(arboles_regresion_limitado, 'Arboles de decision', df_sin_outlier)
    max_depth = max_depth + 1

# Como podemos ver, en el nivel de profundidad 5 tenemos una diferencia de puntuacion cruzada y puntuacion prueba
# aceptable, por no lo que no hay sobreajuste, a mayor profundidad mayor diferencia. Vemos que el error es del 11.8%
# aproximadamente en el dataset sin outliers, es un valor aceptable, el modelo puede hacer buenas predicciones.
# Pero la cantidad de datos es muy baja y el arbol solo puede darnos predicciones aceptables a muy poca profundidad,
# al igual que con la regresion de arboles de decision.
# Por lo que este modelo no es recomendable.

print("\nPREDICCION CON ARBOLES DE DECISION EN DATASET CON OUTLIERS REEMPLAZADOS")
show_fit(arboles_regresion, 'Arboles de decision', df_outlier_reemplazado)

print("\nPREDICCION CON RANDOM FOREST EN DATASET CON OUTLIERS REEMPLAZADOS")
show_fit(arboles_random_regresion, 'Random Forest', df_outlier_reemplazado)

# Con el dataset con outliers reemplazados vemos que tanto con arboles de decision y random forest no se presentan
# sobreajuste y obtienen un error bajo, en arboles de decision tenemos un error del 1.5% por lo que nuestro modelo
# puede predecir casi de forma exacta. Por parte de random forest tenemos que el error es del 3.8%, es un valor un
# poco mas alto que el de arboles de decision aun asi este resultado nos dice que el modelo muy preciso en sus
# predicciones.


# Conclusiones

# Despues de hacer predicciones con estos 4 tipos de regresion: lineal, polinomial, arboles de decision y random forest.
# Podemos concluir que el dataset que mejor resultados da es el que tiene los outliers reemplazados. El problema con el
# dataset normal es que tiene varios valores extremos y eso hace que las predicciones no sean las mejores. EL problema
# con el dataset sin los outliers es la poca cantidad de datos que tiene, hacen que el modelo no pueda hacer buenas
# predicciones con tan poca cantidad de datos. Hay que notar tambien que el dataset con el peor desenpenio en todas las
# regresiones fue el dataset con los outliers eliminados.

# Observamos tambien que para los datos que tenemos en nuestro dataset, la regresion lineal no es la opcion mas
# conveniente teniendo en cuenta los valores de error tan grandes en el dataset sin outliers y el normal, si bien
# con el dataset con outliers reemplazados se pudo dar un resultado mejor, el error sigue siendo mas alto que en las
# otras regresiones. El echo de que se haya dado un buen resultado en regresion lineal puede deberser a la cantidad de
# outliers que fueron reemplazados por el valor de la media de sus respectivas columnas.

# Por parte de la regresion polinomial al querer predecir el rating segun el tipo de anime se ve una mejora en todos los
# dataset. El dataset con mejor rendimiento es el de outliers reemplazados y el dataset con peor rendimiento es el que
# no tiene outliers. Aun asi el error ronda entre el 10%.

# Por parte de los arboles de decision y random forest vemos que hay problemas de sobreajuste en los dataset normal
# y sin outliers, dejando en evidencia que este modelo no funciona bien con el dataset sin outliers por la poca cantidad
# de datos que posee. en el dataset normal hay que ajustarle la profundidad del arbol, aunque con random forest nos da
# una alta precision sin sobreajuste.
# El que mejor resultados dio es el dataset con outliers reemplazados con un error del 1.5% en arboles de decision y un
# 3.8% en random forest. Sin necesidad de modificar el modelo y sin sobreajuste.

# Vemos que la mejor regresion es la de arboles de decision sobre el dataset con outliers reemplazados, dandonos una
# gran precision en nuestras predicciones.
