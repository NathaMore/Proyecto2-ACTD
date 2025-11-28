import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
**LEO EL ARCHIVO DE "LISTINGS.CSV"**




data = pd.read_csv('listings.csv', encoding='latin-1')
**Etapa 2: Extracción, limpieza y transformación**
data.head()
data.info()
Empecamos con 79 columnas y 26401 datos maximos. Podemos apreciar columnas sin datos, varios datos faltantes y varias columnas que no nos sirven para nuestro analísis.
data.price.isnull().sum()
3274 datos faltantes en "price" la variable mas importante.
**Borramos los 3 datos nulos**
data.dropna(subset=["price"], inplace=True)
data.info()
Tenemos datos cuantitativos y cualitativo. Dentro de los cuantitativos tenemos "int64" (enteros) y float64 "pueden tener decimales". Por otro lado, las variables "object" pueden ser tanto cuantitativo o cualitativo dependiendo como escribieron la variable. En caso de que tenga que ser cuantitativo y es cualitativo se puede cambiar o se puede sacar información de ese texto "object" y pasarlo a otra columna númerica como información.

**CHECAMOS DUPLICADOS**
duplicados=data.duplicated().sum()
duplicados
**NO HAY DUPLICADOS**
**BORRAMOS COLUMNAS SIN DATOS**
del data['license']
del data['neighbourhood_group_cleansed']
del data['calendar_updated']
del data['neighbourhood']
data.info()
del data['id']
del data['picture_url']
del data['listing_url']
del data['last_scraped']
del data['host_id']
del data['scrape_id']
del data['host_url']
del data['host_since']
del data['host_location']
del data['host_response_time']
del data['host_response_rate']
del data['host_acceptance_rate']
del data['host_thumbnail_url']
del data['host_picture_url']
del data['host_neighbourhood']
del data['host_verifications']
del data['host_has_profile_pic']
del data['host_identity_verified']
del data['has_availability']
del data['availability_30']
del data['availability_60']
del data['availability_90']
del data['first_review']
del data['last_review']
del data['instant_bookable']
del data['review_scores_accuracy']
del data['review_scores_cleanliness']
del data['review_scores_checkin']
del data['review_scores_communication']
del data['review_scores_location']
del data['review_scores_value']
del data['reviews_per_month']
data.info()
del data['host_is_superhost']
del data['host_listings_count']
del data['minimum_minimum_nights']
del data['maximum_minimum_nights']
del data['minimum_maximum_nights']
del data['maximum_maximum_nights']
data.info()
Para datos faltantes pusimos "Not provided" en columnas específicas.
cols_texto = [
    'description', 'neighborhood_overview', 'host_name',
    'host_about', 'bathrooms_text'
]

for col in cols_texto:
    data[col] = data[col].fillna("Not provided")

data.info()
Extraemos información importante de la columna "bathrooms_text" para poder completar datos faltantes en la columna "bathrooms".
import re

def extract_bathrooms(text):
    if pd.isna(text):
        return None
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return None

data['bathrooms_parsed'] = data['bathrooms_text'].apply(extract_bathrooms)

# Usar bathrooms_parsed para rellenar bathrooms
data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms_parsed'])

# Si todavía quedan NaN → llenar con mediana
data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms'].median())

data.drop(columns=['bathrooms_parsed'], inplace=True)

Llenamos otros datos daltantes con diferentes técnicas. Por ejemplo: Con la variable "bedrooms" y "beds"llenamos con la mediana.
data['bedrooms'] = data.groupby('neighbourhood_cleansed')['bedrooms'].transform(
    lambda x: x.fillna(x.median())
)

data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())

data['beds'] = data['beds'].fillna(data['bedrooms'])
data['beds'] = data['beds'].fillna(data['beds'].median())

data['host_total_listings_count'] = data['host_total_listings_count'].fillna(
    data['host_total_listings_count'].median()
)

data['review_scores_rating'] = data.groupby('neighbourhood_cleansed')['review_scores_rating'].transform(
    lambda x: x.fillna(x.median())
)

data['review_scores_rating'] = data['review_scores_rating'].fillna(
    data['review_scores_rating'].median()
)

data['calendar_last_scraped'] = pd.to_datetime(data['calendar_last_scraped'])

data.isna().sum()
data["bathrooms"].value_counts()
data.info()
data["calendar_last_scraped"].value_counts()
En esta etapa se limpió la base de datos por completo, dejando todas las variables con el mismo número de datos. Se borraron las columnas innecesarias y se llenaron datos faltantes mediante la mediana o información de otra columna (como en bathrooms).Todas las variables estan listas para modelar una regresión y realizar gráficas.
**Etapa** 3

data.shape
data.describe()
Apreciamos cuantos datos tienen las variables,la media, el mínimo, el máximo y los cuartiles de cada variable.
data.dtypes
data.isna().sum()
Convertimos el precio a una variable cuantitativa, quitandole signos y comas que no permiten que sea cuantitativa.
# Limpiar y convertir price a float
data["price"] = (
    data["price"].astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Checar rápido
data["price"].head()
data["price"].describe()
import matplotlib.pyplot as plt
import seaborn as sns

variables = ["price", "accommodates", "bedrooms", "beds",
             "bathrooms", "number_of_reviews", "review_scores_rating"]

plt.figure(figsize=(14,10))
for i, var in enumerate(variables, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[var], kde=True, bins=30)
    plt.title(f"Histograma de {var}")
plt.tight_layout()
plt.show()

Diferentes histogramas que demuestran el comportamiento de diferentes variables. Podemos apreciar que el único que tiene un sesgo a la izquierda es la variable "review_scores_rating" por otro lado, todas las otras variables muestran un sesgo a la derecha.
plt.figure(figsize=(6,5))
sns.boxplot(x=data["price"])
plt.title("Boxplot de Price")
plt.show()

El comportamiento del precio esta mayormente por debajo de 200,000 con dos outliers pasando 600,000.
plt.figure(figsize=(8,6))
sns.boxplot(x="room_type", y="price", data=data)
plt.title("Precios por tipo de habitación")
plt.xticks(rotation=45)
plt.show()

Con estas gráficas boxplot podemos apreciar que todas menos "shared room" tienen outliers. La variable "Entire home/ apt" cuanta con varios outliers pero todos estan muy cerca estando por debajo de la marca de 200,000. Por otro lado, las variables de "private room" y "hotel room" cuantan con al menos un outlier que ronda por arriba de los 600,000.
plt.figure(figsize=(6,5))
sns.scatterplot(x="accommodates", y="price", data=data)
plt.title("Price vs Accommodates")
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(x="bedrooms", y="price", data=data)
plt.title("Price vs Bedrooms")
plt.show()

Estos diagramas de dispersión son muy útiles para ver la relación entre dos variables. Con el primero podemos apreciar que no necesariamente crece el precio conforme van creciendo los "accommodates" aunque pareciera que al final con 16 "accommodates" da un pequeño salto.
En cuanto a la relación de bedrooms y price, apreciamos que con la mayoría de los datos acaba con un máximo de casi 30 "bedrooms" si se puede apreciar una leve subida e precio dependiendo la cantidad de camas pero igulmente con datos menores a 10 camas se pueden ver precios incluso mas altos a los que tienen mas de 15 camas.
data["price"].value_counts()
data["price"].dtype

Q1 = data["price"].quantile(0.25)
Q3 = data["price"].quantile(0.75)
IQR = Q3 - Q1

lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

data = data[(data["price"] >= lim_inf) & (data["price"] <= lim_sup)]
print(lim_inf)
print(lim_sup)
Para mejorar el modelo, eliminé valores atípicos del precio usando el método IQR (Rango Intercuartílico)
Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


numeric_vars = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating"
]

categorical_vars = [
    "property_type",
    "room_type",
    "neighbourhood_cleansed"
]
corr = data[numeric_vars + ["price"]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()

Mediante este diagrama podemos apreciar que variables podrían ser utiles para la regresión lineal. Normalmente las que tienen un valor mas alto sin importar el signo podrían ayudar a crear una buena regresión lineal. Como estamos haciendo para precio, acommodates, bedrooms, beds y bathrooms sí ayudan al modelo.
numeric_vars = [c for c in numeric_vars if c in data.columns]
categorical_vars = [c for c in categorical_vars if c in data.columns]

all_vars = numeric_vars + categorical_vars
df = data[["price"] + all_vars].dropna()

X = df[all_vars]
y = df["price"]
X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=42
)
La regresión se entrena con 70% de los datos y se prueba con el 30% faltante.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
**Modelo creado**
import scipy.stats as stats
import matplotlib.pyplot as plt

# Calcular residuales
residuals = y_test - y_pred

# Q-Q Plot
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot de Residuales")
plt.show()

Gráfica que demuestra el comportamieno de nuestro modelo.
print("R²:", r2)
Con nuestro modelo se obtuvo una R² con los diferentes variables numpericos y categóricos.

Variables númericos:


*   "accommodates"
*   "bedrooms"
*   "beds"
*   "bathrooms"
*   "minimum_nights"
*   "availability_365"
*   "number_of_reviews"
*   "review_scores_rating"

Variables categóricos:

*   "property_type"
*   "room_type"
*   "neighbourhood_cleansed"








# Intercepto
b0 = model.intercept_
print("b0 (intercept):", b0)

# Coeficientes
print("Coefficients from model.coef_:")
print(model.coef_)

# Número de coeficientes
print("Number of coefficients:", len(model.coef_))

# Coeficientes con nombres de variables
coef_table = pd.DataFrame({
    "variable": X.columns,
    "coef": model.coef_
})

print("\nCoeficientes por variable:")
print(coef_table)



residuals = y_test - y_pred

**Histograma de residuales del modelo.**
plt.figure(figsize=(7,5))
sns.histplot(residuals, bins=15, kde=True)
plt.title('Histograma de Residuales del Modelo')
plt.xlabel('Error (y_test - y_pred)')
plt.ylabel('Frecuencia')
plt.show()

ax1 = sns.kdeplot(y_test, color="r", label="Actual")
sns.kdeplot(y_pred, color="b", label="Predicted", ax=ax1)

plt.title("Distribución Real vs Predicha")
plt.xlabel("Precio")
plt.ylabel("Densidad")
plt.legend()
plt.show()

El modelo de regresión lineal logró explicar una parte  de la variabilidad del precio, especialmente después de eliminar mediante IQR, lo que mejoró el R² a aproximadamente 0.40. Aun presenta limitaciones debido a la dispersión del precio en Airbnb. Aun así, el modelo funciona como una herramienta inicial para estimar precios y entender qué variables tienen mayor impacto en ellos.
## Modelo predictivo Uniandes
### Pregunta 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def entrenar_modelo(neuronas_capa1, neuronas_capa2, learning_rate,
                    epochs, batch_size, activacion, nombre_run):

    # Definir el modelo
    model = Sequential([
        Dense(neuronas_capa1, activation=activacion, input_shape=(X_train_scaled.shape[1],)),
        Dense(neuronas_capa2, activation=activacion),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    print(f"Entrenando: {nombre_run}")

    # Entrenamiento
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    # Evaluación
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✔ Test MAE: {test_mae:.2f}")

    return model, test_mae
#Experimentos a realizar
lista_experimentos = [
    (64, 32, 0.001, 50, 32, "relu",  "NN_relu_64x32"),
    (64, 32, 0.001, 50, 32, "tanh",  "NN_tanh_64x32"),
    (128, 64, 0.001, 50, 32, "relu", "NN_relu_128x64"),
    (128, 64, 0.001, 50, 32, "selu", "NN_selu_128x64"),
    (128, 64, 0.0005, 80, 32, "relu", "NN_relu_128x64_lento"),
]
#Evaluar experimentos y seleccionar el mejor modelo
mejor_mae = np.inf
mejor_modelo = None
mejor_config = None

for n1, n2, lr, ep, bs, act, name in lista_experimentos:
    modelo_tmp, mae_tmp = entrenar_modelo(n1, n2, lr, ep, bs, act, name)

    if mae_tmp < mejor_mae:
        mejor_mae = mae_tmp
        mejor_modelo = modelo_tmp
        mejor_config = name

print(f"Mejor modelo: {mejor_config} con MAE = {mejor_mae:.2f}")

### Pregunta 2
# Seleccionar un departamento base (representativo)
base_df = X_train.iloc[[0]].copy()

def predecir(df_sim):
    X_sim_scaled = scaler.transform(df_sim[X_train.columns])
    return mejor_modelo.predict(X_sim_scaled)[0][0]

precio_base = predecir(base_df)
print(f"Precio base predicho: {precio_base:.2f}")
#### Analisis sensibildiad puntuacion
resultados_rating = []

for r in np.linspace(3.0, 5.0, 11):  # rango REAL de rating
    sim = base_df.copy()
    sim["review_scores_rating"] = r
    precio = predecir(sim)
    resultados_rating.append({"rating": r, "precio": precio})

df_rating = pd.DataFrame(resultados_rating)
display(df_rating)

plt.plot(df_rating["rating"], df_rating["precio"], marker="o")
plt.title("Precio predicho vs Rating")
plt.xlabel("review_scores_rating")
plt.ylabel("Precio predicho")
plt.grid(True)
plt.show()
#### Sensibilidad numero de huespedes
resultados_acc = []

for acc in range(1, 16):
    sim = base_df.copy()
    sim["accommodates"] = acc
    precio = predecir(sim)
    resultados_acc.append({"accommodates": acc, "precio": precio})

df_acc = pd.DataFrame(resultados_acc)
display(df_acc)

plt.plot(df_acc["accommodates"], df_acc["precio"], marker="o")
plt.title("Precio predicho vs Accommodates")
plt.xlabel("Número de huéspedes")
plt.ylabel("Precio predicho")
plt.grid(True)
plt.show()

#### Sensibilidad tipo de habitación
resultados_room = []

categorias = X_train.filter(like="room_type_").columns

for cat in categorias:
    sim = base_df.copy()
    sim[categorias] = 0  # poner todas en 0
    sim[cat] = 1         # activar solo una
    precio = predecir(sim)
    resultados_room.append({"room_type": cat, "precio": precio})

df_room = pd.DataFrame(resultados_room)
display(df_room)

#### Sensibilidad numero de habitaciones
resultados_bedrooms = []

for br in range(0, 7):
    sim = base_df.copy()
    sim["bedrooms"] = br
    precio = predecir(sim)
    resultados_bedrooms.append({"bedrooms": br, "precio": precio})

df_bedrooms = pd.DataFrame(resultados_bedrooms)
display(df_bedrooms)

plt.plot(df_bedrooms["bedrooms"], df_bedrooms["precio"], marker="o")
plt.title("Precio predicho vs Bedrooms")
plt.xlabel("Habitaciones (bedrooms)")
plt.ylabel("Precio predicho")
plt.grid(True)
plt.show()
#### Sensibilidad numero de baños
resultados_bathrooms = []

for b in range(0, 15):  # 1.0, 1.5, 2.0, ..., 4.0
    sim = base_df.copy()
    sim["bathrooms"] = b
    precio = predecir(sim)
    resultados_bathrooms.append({"bathrooms": b, "precio": precio})

df_bath = pd.DataFrame(resultados_bathrooms)
display(df_bath)

plt.plot(df_bath["bathrooms"], df_bath["precio"], marker="o")
plt.title("Precio predicho vs Bathrooms")
plt.xlabel("Baños")
plt.ylabel("Precio predicho")
plt.grid(True)
plt.show()

### pregunta 3
# y sigue siendo el precio real (de tu modelo de regresión)
umbral_precio = y.median()
print("Umbral de precio para ser recomendado:", umbral_precio)

# Variable binaria: 1 = recomendado, 0 = no recomendado
y_clas = (y > umbral_precio).astype(int)
y_clas.value_counts()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clas, test_size=0.3, random_state=42, stratify=y_clas
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)
def entrenar_modelo_clas(neuronas_capa1, neuronas_capa2, learning_rate,
                         epochs, batch_size, activacion, nombre_run):

    n_features_clf = X_train_clf_scaled.shape[1]

    # Definir el modelo de clasificación
    model = Sequential([
        Dense(neuronas_capa1, activation=activacion, input_shape=(n_features_clf,)),
        Dense(neuronas_capa2, activation=activacion),
        Dense(1, activation='sigmoid')       # salida binaria
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamiento
    history = model.fit(
        X_train_clf_scaled, y_train_clf,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    # Evaluación
    test_loss, test_acc = model.evaluate(X_test_clf_scaled, y_test_clf, verbose=0)
    print(f"✔ Accuracy en test: {test_acc:.3f}")

    return model, test_acc

# Lista de experimentos
lista_experimentos_clf = [
    (64, 32, 0.001, 50, 32, "relu", "CLF_relu_64x32"),
    (64, 32, 0.001, 50, 32, "tanh", "CLF_tanh_64x32"),
    (128, 64, 0.001, 50, 32, "relu", "CLF_relu_128x64"),
]

# Selección del mejor modelo
mejor_acc = 0
mejor_modelo_clf = None
mejor_conf_clf = None

for n1, n2, lr, ep, bs, act, name in lista_experimentos_clf:
    modelo_tmp, acc_tmp = entrenar_modelo_clas(n1, n2, lr, ep, bs, act, name)

    if acc_tmp > mejor_acc:
        mejor_acc = acc_tmp
        mejor_modelo_clf = modelo_tmp
        mejor_conf_clf = name

print(f"Mejor clasificador: {mejor_conf_clf} con accuracy = {mejor_acc:.3f}")
X_all_scaled_clf = scaler_clf.transform(X)
prob_recomendado = mejor_modelo_clf.predict(X_all_scaled_clf).flatten()
labels_recomendado = (prob_recomendado >= 0.5).astype(int)

df_result_clf = pd.DataFrame({
    "precio_real": y.values,
    "prob_recomendado": prob_recomendado,
    "recomendado": labels_recomendado
})

df_result_clf.head()