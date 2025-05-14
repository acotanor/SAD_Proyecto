import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Cargar los datos
def cargar_datos(csv_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    # Verificar que las columnas necesarias existan
    if 'Review' not in df.columns or 'Positive or Negative' not in df.columns:
        raise ValueError("El archivo CSV debe contener las columnas 'Review' y 'Positive or Negative'.")
    # Eliminar filas con valores NaN en las columnas necesarias
    df = df.dropna(subset=['Review', 'Positive or Negative'])
    return df

# 2. Preprocesar el texto
def preprocesar_texto(texto):
    # Eliminar caracteres especiales y convertir a minúsculas
    texto = re.sub(r'[^a-zA-Z\s]', '', texto).lower()
    return texto

# 3. Entrenar y probar varios valores de k
def entrenar_knn_con_varios_k(csv_path, k_values):
    # Cargar los datos
    df = cargar_datos(csv_path)

    # Preprocesar los comentarios
    df['Review'] = df['Review'].fillna("").apply(preprocesar_texto)

    # Dividir en características (X) y etiquetas (y)
    X = df['Review']
    y = df['Positive or Negative']

    # Convertir texto en vectores TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Limitar a las 5000 palabras más frecuentes
    X_tfidf = vectorizer.fit_transform(X)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Probar diferentes valores de k
    mejor_k = None
    mejor_accuracy = 0
    for k in k_values:
        print(f"Probando k={k}...")
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Exactitud para k={k}: {accuracy:.4f}")

        # Guardar el mejor k
        if accuracy > mejor_accuracy:
            mejor_k = k
            mejor_accuracy = accuracy

    print(f"\nMejor valor de k: {mejor_k} con exactitud: {mejor_accuracy:.4f}")

    # Entrenar el modelo final con el mejor k
    model_final = KNeighborsClassifier(n_neighbors=mejor_k)
    model_final.fit(X_train, y_train)

    # Evaluar el modelo final
    y_pred_final = model_final.predict(X_test)
    print("\nReporte de clasificación para el mejor modelo:")
    print(classification_report(y_test, y_pred_final))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred_final))

    return model_final, vectorizer

# 4. Guardar el modelo entrenado
def guardar_modelo(model, vectorizer, model_path, vectorizer_path):
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Modelo guardado en: {model_path}")
    print(f"Vectorizador guardado en: {vectorizer_path}")

# 5. Ejecutar el entrenamiento
if __name__ == "__main__":
    # Ruta del archivo CSV
    csv_path = "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\AirBNBReviews.csv"

    # Valores de k a probar
    k_values = [1, 3, 5, 7, 9, 11]

    # Entrenar el modelo con varios valores de k
    modelo, vectorizador = entrenar_knn_con_varios_k(csv_path, k_values)

    # Guardar el modelo y el vectorizador
    guardar_modelo(modelo, vectorizador,
                   "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\modelo_knn.pkl",
                   "c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\vectorizador_tfidf.pkl")