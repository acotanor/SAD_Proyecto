import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
import joblib

# Descargar el lexicón de VADER
nltk.download('vader_lexicon')

def entrenar_y_predecir(input_csv, output_csv, modelo_path="modelo_random_forest.pkl"):
    """
    Entrena un modelo supervisado usando los puntajes de VADER como características
    y predice las puntuaciones de los comentarios en una escala del 1 al 9.

    :param input_csv: Ruta del archivo CSV de entrada con las columnas 'Review' y 'Rating'.
    :param output_csv: Ruta del archivo CSV de salida con las predicciones.
    :param modelo_path: Ruta del archivo donde se guardará el modelo entrenado.
    """
    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Verificar que las columnas 'Review' y 'Rating' existan
    if 'Review' not in df.columns or 'Rating' not in df.columns:
        raise ValueError("El archivo de entrada debe contener las columnas 'Review' y 'Rating'.")

    # Escalar las etiquetas originales a la escala del 1 al 9
    df['Scaled_Rating'] = (df['Rating'] * 2.25).round().clip(1, 9)

    # Inicializar el analizador de sentimientos
    sia = SentimentIntensityAnalyzer()

    # Calcular los puntajes de VADER para cada comentario
    def calcular_puntajes_vader(review):
        if pd.isna(review):
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        return sia.polarity_scores(review)

    vader_scores = df['Review'].fillna("").apply(calcular_puntajes_vader)
    vader_df = pd.DataFrame(vader_scores.tolist())

    # Combinar los puntajes de VADER con las puntuaciones originales
    df = pd.concat([df, vader_df], axis=1)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = df[['compound', 'pos', 'neu', 'neg']]
    y = df['Scaled_Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo supervisado (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, modelo_path)
    print(f"Modelo guardado en: {modelo_path}")

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Predecir las puntuaciones para todos los comentarios
    df['Predicted_Rating'] = model.predict(X)

    # Asegurar que las predicciones estén en la escala del 1 al 9
    df['Predicted_Rating'] = df['Predicted_Rating'].clip(1, 9)

    # Guardar el archivo con las predicciones
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

def predecir_nuevos_comentarios(input_csv, output_csv, modelo_path="modelo_random_forest.pkl"):
    """
    Aplica un modelo entrenado a un nuevo conjunto de datos para predecir puntuaciones.

    :param input_csv: Ruta del archivo CSV de entrada con la columna 'Review'.
    :param output_csv: Ruta del archivo CSV de salida con las predicciones.
    :param modelo_path: Ruta del archivo del modelo guardado.
    """
    # Cargar el modelo entrenado
    model = joblib.load(modelo_path)
    print("Modelo cargado desde:", modelo_path)

    # Leer el archivo CSV de entrada
    df = pd.read_csv(input_csv)

    # Verificar que la columna 'Review' exista
    if 'Review' not in df.columns:
        raise ValueError("El archivo de entrada debe contener la columna 'Review'.")

    # Inicializar el analizador de sentimientos
    sia = SentimentIntensityAnalyzer()

    # Calcular los puntajes de VADER para cada comentario
    def calcular_puntajes_vader(review):
        if pd.isna(review):
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        return sia.polarity_scores(review)

    vader_scores = df['Review'].fillna("").apply(calcular_puntajes_vader)
    vader_df = pd.DataFrame(vader_scores.tolist())

    # Combinar los puntajes de VADER con los datos originales
    df = pd.concat([df, vader_df], axis=1)

    # Seleccionar las características para la predicción
    X = df[['compound', 'pos', 'neu', 'neg']]

    # Predecir las puntuaciones para los nuevos comentarios
    df['Predicted_Rating'] = model.predict(X)

    # Asegurar que las predicciones estén en la escala del 1 al 9
    df['Predicted_Rating'] = df['Predicted_Rating'].clip(1, 9)

    # Guardar el archivo con las predicciones
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo CSV de entrada
    input_csv = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\tripadvisor_hotel_reviews.csv"
    
    # Ruta del archivo CSV de salida
    output_csv = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\predicciones_comentarios.csv"
    
    # Llamar a la función de entrenamiento y predicción
    entrenar_y_predecir(input_csv, output_csv)

    # Ruta del archivo CSV de entrada para nuevos comentarios
    input_csv_nuevos = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\nuevo_csv_comentarios.csv"
    
    # Ruta del archivo CSV de salida para nuevos comentarios
    output_csv_nuevos = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\predicciones_nuevos_comentarios.csv"
    
    # Aplicar el modelo a los nuevos comentarios
    predecir_nuevos_comentarios(input_csv_nuevos, output_csv_nuevos)