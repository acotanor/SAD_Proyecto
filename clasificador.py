# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""
import chardet
from nltk import SnowballStemmer
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB  # Añade esto con los otros imports de sklearn
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# Funciones auxiliares
def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)",
                       required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator",
                       help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter",
                       required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,
                       type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False,
                       action="store_true")
    parse.add_argument("--debug",
                       help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]",
                       required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()

    # Leemos los parametros del JSON
    with open('config.json') as json_file:
        config = json.load(json_file)

    #Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)

    # Parseamos los argumentos
    return args


def load_data(file):
    try:
        # Primero intenta con UTF-8
        try:
            data = pd.read_csv(file, encoding='utf-8', delimiter=',', quotechar='"', on_bad_lines='skip')
        except UnicodeDecodeError:
            # Si falla, prueba con codificaciones comunes
            encodings = ['windows-1252', 'iso-8859-1', 'latin1']
            for enc in encodings:
                try:
                    data = pd.read_csv(file, encoding=enc, delimiter=';', quotechar='"', on_bad_lines='skip')
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("No se pudo determinar la codificación del archivo")


        if data.empty:
            raise ValueError("El archivo CSV está vacío")
        if args.prediction not in data.columns:
            print(data.columns)
            raise ValueError(f"Columna objetivo '{args.prediction}' no encontrada")

        print(Fore.GREEN + f"Datos cargados con éxito. Forma: {data.shape}" + Fore.RESET)
        return data

    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para calcular métricas
def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro


def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


def calculate_classification_report(y_test, y_pred):
    """
    Función para calcular el informe de clasificación
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Informe de clasificación
    """
    report = classification_report(y_test, y_pred, zero_division=0)
    return report


# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:,
                              categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Columnas numéricas:\n" + Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA + "> Columnas de texto:\n" + Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA + "> Columnas categóricas:\n" + Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        # Primero verifica si hay columnas con valores faltantes
        cols_with_missing = data.columns[data.isnull().any()].tolist()
        if not cols_with_missing:
            print(Fore.YELLOW + "No hay valores faltantes para tratar" + Fore.RESET)
            return

        print(Fore.YELLOW + f"Columnas con valores faltantes: {cols_with_missing}" + Fore.RESET)

        # Verifica la estrategia configurada
        strategy = args.preprocessing.get("missing_values", None)

        if strategy == "drop":
            # Elimina filas con valores faltantes solo en las columnas relevantes
            relevant_cols = list(numerical_feature.columns) + list(categorical_feature.columns)
            cols_to_drop = [col for col in cols_with_missing if col in relevant_cols]

            if cols_to_drop:
                data.dropna(subset=cols_to_drop, inplace=True)
                print(Fore.GREEN + f"Missing values eliminados en columnas: {cols_to_drop}" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No hay valores faltantes en columnas numéricas o categóricas" + Fore.RESET)

        elif strategy == "impute":
            impute_strategy = args.preprocessing.get("impute_strategy", "mean")

            # Imputación para columnas numéricas
            num_cols = [col for col in numerical_feature.columns if col in cols_with_missing]
            if num_cols:
                if impute_strategy == "mean":
                    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
                elif impute_strategy == "median":
                    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
                elif impute_strategy == "most_frequent":
                    data[num_cols] = data[num_cols].fillna(data[num_cols].mode().iloc[0])
                print(
                    Fore.GREEN + f"Missing values imputados en columnas numéricas {num_cols} usando {impute_strategy}" + Fore.RESET)

            # Imputación para columnas categóricas
            cat_cols = [col for col in categorical_feature.columns if col in cols_with_missing]
            if cat_cols:
                if impute_strategy == "most_frequent":
                    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
                else:
                    # Para categóricas solo usamos "most_frequent" aunque esté configurado otro
                    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
                    print(
                        Fore.GREEN + f"Missing values imputados en columnas categóricas {cat_cols} usando moda" + Fore.RESET)

        else:
            print(Fore.YELLOW + "No se están tratando los missing values" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al tratar los missing values" + Fore.RESET)
        print(e)
        sys.exit(1)


# TODO aqui lo que hayais hecho

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    global data
    try:
        if numerical_feature.columns.size > 0:
            if args.preprocessing["scaling"] == "standard":
                scaler = StandardScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados usando StandardScaler" + Fore.RESET)
            elif args.preprocessing["scaling"] == "minmax":
                scaler = MinMaxScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados usando MinMaxScaler" + Fore.RESET)
            elif args.preprocessing["scaling"] == "maxabs":
                scaler = MaxAbsScaler()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados usando MaxAbsScaler" + Fore.RESET)
            elif args.preprocessing["scaling"] == "normalizer":
                scaler = Normalizer()
                data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos reescalados usando Normalizer" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están escalando los datos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas numéricas" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


# TODO aqui reescalar

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global data
    try:
        if categorical_feature.columns.size > 0:
            labelencoder = LabelEncoder()
            for col in categorical_feature.columns:
                data[col] = labelencoder.fit_transform(data[col])
                print(Fore.GREEN + "Datos categóricos pasados a numéricos con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas categóricas que pasar a numericas" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al pasar los datos categóricos a numéricos" + Fore.RESET)
        print(e)
        sys.exit(1)


# TODO aqui lo que haga falta para pasar de categorial a numerico

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....

    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.

    Retorna:
    None
    """
    global data
    try:
        if text_feature.columns.size > 0:
            stop_words = set(stopwords.words('english'))
            stemmer = SnowballStemmer('english')
            for col in text_feature.columns:
                data[col] = data[col].apply(lambda x: ' '.join(sorted(
                    [stemmer.stem(word) for word in word_tokenize(x.lower()) if
                     word not in stop_words and word not in string.punctuation])))
            print(Fore.GREEN + "Texto simplificado" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a simplificar" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


# TODO aqui lo que sea preciso en caso de tener texto

def process_text(text_feature):
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":
                # Configuración optimizada (5 parámetros clave)
                vectorizer = TfidfVectorizer(
                    max_features=args.preprocessing.get("max_text_features", 3000),
                    min_df=args.preprocessing.get("text_min_df", 5),
                    max_df=args.preprocessing.get("text_max_df", 0.75),
                    ngram_range=(1, 2),  # Unigramas y bigramas
                    sublinear_tf=True  # Suavizado logarítmico
                )

                # Procesamiento eficiente
                text_data = data[text_feature.columns[0]].fillna('')
                tfidf_matrix = vectorizer.fit_transform(text_data)

                # Convertir a DataFrame sin consumir toda la RAM
                text_features = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
                text_features.columns = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]

                # Debug: Mostrar términos clave
                print(Fore.CYAN + "\nTop 10 términos TF-IDF:" + Fore.RESET)
                print(vectorizer.get_feature_names_out()[:10])

                data = pd.concat([data, text_features], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)

                print(Fore.GREEN + f"Texto vectorizado. Features: {tfidf_matrix.shape[1]}" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error en process_text:" + Fore.RESET, str(e))
        sys.exit(1)


def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    global data
    if args.mode != "test":
        try:
            if args.preprocessing["sampling"] == "oversampling":
                ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = ros.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN + "Oversampling realizado" + Fore.RESET)
            elif args.preprocessing["sampling"] == "undersampling":
                rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = rus.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN + "Undersampling realizado" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están realizando oversampling o undersampling" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al realizar oversampling o undersampling" + Fore.RESET)
            print(e)
            sys.exit(1)
    else:
        print(Fore.GREEN + "No se realiza oversampling o undersampling en modo test" + Fore.RESET)


def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
        print(e)
        sys.exit(1)


def preprocesar_datos():
    global data
    try:
        # 1. Separar características
        numerical_feature, text_feature, categorical_feature = select_features()

        # 2. Procesar texto (¡Ahora optimizado!)
        simplify_text(text_feature)  # Limpieza básica
        process_text(text_feature)  # Vectorización TF-IDF

        # 3. Otras transformaciones
        cat2num(categorical_feature)
        process_missing_values(numerical_feature, categorical_feature)
        reescaler(numerical_feature)
        over_under_sampling()

        print(Fore.GREEN + "Preprocesamiento completado." + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error en preprocesar_datos:" + Fore.RESET, str(e))
        sys.exit(1)


# Funciones para entrenar un modelo

def divide_data_three_ways():
    """
    Divide los datos en train-dev-test según las proporciones del config.json.
    """
    y = data[args.prediction]
    x = data.drop(columns=[args.prediction])

    # Primera división: train (60%) vs temp (40%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y,
        test_size=(args.data_split["dev_ratio"] + args.data_split["test_ratio"]),
        random_state=args.data_split["random_state"]
    )

    # Segunda división: dev (20%) y test (20%) del temp (40%)
    dev_ratio = args.data_split["dev_ratio"] / (args.data_split["dev_ratio"] + args.data_split["test_ratio"])
    x_dev, x_test, y_dev, y_test = train_test_split(
        x_temp, y_temp,
        test_size=args.data_split["test_ratio"] / (args.data_split["dev_ratio"] + args.data_split["test_ratio"]),
        random_state=args.data_split["random_state"]
    )

    return x_train, x_dev, x_test, y_train, y_dev, y_test


def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        # Crear directorio con timestamp
        model_dir = f"output/model_{args.mode}_{args.algorithm}"
        os.makedirs(model_dir, exist_ok=True)

        # Guardar modelo
        model_path = os.path.join(model_dir, 'modelo.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(gs, file)

        # Guardar métricas y parámetros
        with open(os.path.join(model_dir, 'metrics.csv'), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Parametro', 'Valor'])
            writer.writerow(['Best Score', gs.best_score_])
            for param, value in gs.best_params_.items():
                writer.writerow([param, value])

        print(Fore.CYAN + f"Modelo guardado en {model_dir}" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)


def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA + "> Mejores parametros:\n" + Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA + "> Mejor puntuacion:\n" + Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET,
              calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET,
              calculate_confusion_matrix(y_dev, gs.predict(x_dev)))


def kNN():
    # Dividimos en train-dev-test
    x_train, x_dev, x_test, y_train, y_dev, y_test = divide_data_three_ways()

    # Entrenamiento con GridSearchCV (usa train y dev para validación cruzada)
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        gs.fit(x_train, y_train)
        # [Simulación de barra de progreso...]

    # Evaluación en dev (opcional, para ajuste de hiperparámetros)
    print("\nResultados en Dev Set:")
    mostrar_resultados(gs, x_dev, y_dev)

    # Evaluación FINAL en test (nuevo)
    print("\n\n" + Fore.CYAN + "=== Evaluación en Test Set ===" + Fore.RESET)
    test_pred = gs.predict(x_test)
    print(Fore.MAGENTA + "> F1-score (test):\n" + Fore.RESET, f1_score(y_test, test_pred, average='macro'))
    print(Fore.MAGENTA + "> Matriz de confusión (test):\n" + Fore.RESET, confusion_matrix(y_test, test_pred))

    # Guardar modelo y resultados
    save_model(gs)


def decision_tree():
    """
    Entrena un modelo de árbol de decisión con GridSearchCV y evalúa en dev/test.
    """
    # Dividimos en train-dev-test (60-20-20)
    x_train, x_dev, x_test, y_train, y_dev, y_test = divide_data_three_ways()

    # Entrenamiento con GridSearchCV
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid=args.decision_tree,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )
        gs.fit(x_train, y_train)

        # Simulación de barra de progreso (opcional)
        for _ in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(1)

    # Evaluación en dev set
    print("\nResultados en Dev Set:")
    mostrar_resultados(gs, x_dev, y_dev)

    # Evaluación FINAL en test set (¡NUEVO!)
    print("\n\n" + Fore.CYAN + "=== Evaluación en Test Set ===" + Fore.RESET)
    test_pred = gs.predict(x_test)
    print(Fore.MAGENTA + "> F1-score (test):\n" + Fore.RESET,
          f1_score(y_test, test_pred, average='macro'))
    print(Fore.MAGENTA + "> Matriz de confusión (test):\n" + Fore.RESET,
          confusion_matrix(y_test, test_pred))
    print(Fore.MAGENTA + "> Informe de clasificación (test):\n" + Fore.RESET,
          classification_report(y_test, test_pred, zero_division=0))

    # Guardar modelo y resultados
    save_model(gs)


def random_forest():
    # Dividimos en train-dev-test
    x_train, x_dev, x_test, y_train, y_dev, y_test = divide_data_three_ways()

    # Entrenamiento con GridSearchCV
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        gs.fit(x_train, y_train)

    # Evaluación en dev y test
    print("\nResultados en Dev Set:")
    mostrar_resultados(gs, x_dev, y_dev)

    print("\n\n" + Fore.CYAN + "=== Evaluación en Test Set ===" + Fore.RESET)
    test_pred = gs.predict(x_test)
    print(Fore.MAGENTA + "> F1-score (test):\n" + Fore.RESET, f1_score(y_test, test_pred, average='macro'))
    print(Fore.MAGENTA + "> Matriz de confusión (test):\n" + Fore.RESET, confusion_matrix(y_test, test_pred))

    save_model(gs)


def naive_bayes():
    """
    Entrena un modelo Naive-Bayes con GridSearchCV y evalúa en dev/test.
    """
    # Dividimos en train-dev-test (60-20-20)
    x_train, x_dev, x_test, y_train, y_dev, y_test = divide_data_three_ways()

    # Entrenamiento con GridSearchCV
    with tqdm(total=100, desc='Procesando Naive-Bayes', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(
            MultinomialNB(),
            param_grid=args.naive_bayes,
            cv=5,
            n_jobs=args.cpu,
            scoring=args.estimator
        )
        gs.fit(x_train, y_train)

        # Simulación de barra de progreso
        for _ in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(1)

    # Evaluación en dev set
    print("\nResultados en Dev Set:")
    mostrar_resultados(gs, x_dev, y_dev)

    # Evaluación FINAL en test set
    print("\n\n" + Fore.CYAN + "=== Evaluación en Test Set ===" + Fore.RESET)
    test_pred = gs.predict(x_test)
    print(Fore.MAGENTA + "> F1-score (test):\n" + Fore.RESET,
          f1_score(y_test, test_pred, average='macro'))
    print(Fore.MAGENTA + "> Matriz de confusión (test):\n" + Fore.RESET,
          confusion_matrix(y_test, test_pred))
    print(Fore.MAGENTA + "> Informe de clasificación (test):\n" + Fore.RESET,
          classification_report(y_test, test_pred, zero_division=0))

    # Guardar modelo y resultados
    save_model(gs)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(f'output/model_train_{args.algorithm}/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)


def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    try:
        # Primero eliminamos la columna objetivo si existe
        if args.prediction in data.columns:
            X_pred = data.drop(columns=[args.prediction])
        else:
            X_pred = data.copy()

        # Predecimos solo con las características
        prediction = model.predict(X_pred)

        # Añadimos la predicción al dataframe original
        data[args.prediction + '_pred'] = prediction
        print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)

        # Guardamos los resultados
        data.to_csv(f'output/model_train_{args.algorithm}/data-prediction.csv', index=False)
        print(Fore.GREEN + "Resultados guardados en output/data-prediction.csv" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error durante la predicción" + Fore.RESET)
        print(e)
        sys.exit(1)


# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN + "La carpeta output ya existe" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    print("Columnas disponibles:", list(data.columns))
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN + "Algoritmo kNN ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN + "Algoritmo árbol de decisión ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN + "Algoritmo random forest ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":  # Nuevo caso
            try:
                naive_bayes()
                print(Fore.GREEN + "Algoritmo Naive-Bayes ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED + "Modo no soportado" + Fore.RESET)
        sys.exit(1)
