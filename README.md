# Requisitos de uso:
Para instalar los requirements hay que ejecutar los siguientes comandos:

1. python3 -m venv /ruta/del/entorno/.venv
2. source /ruta/del/entorno/.venv/bin/activate
3. python3 -m pip install --upgrade pip
4. pip install -r requirements.txt


# Entrenar modelos:

¡La carpeta modelos ya contiene modelos entrenados!
## Clasificador binario knn:
    python3 entrenar_pos_neg.py -i Datos/AirBNBReviews.csv

## Clasificador categorial random forest:
    python3 entrenar_TripAdvisor.py -i Datos/tripadvisor_hotel_reviews.csv

## Clasificador BERT:
    python3 BERT.py -i Datos/tripadvisor_hotel_reviews.csv

# Evaluar Comentarios:

## Evaluar todos los comentarios de un archivo csv:
    python3 predecirComentarioBert.py -i Datos/AirBNBReviews.csv -o Datos/AirBNBReviews_evaluado.csv  

## Evaluar comentarios individuales (escritos desde la terminal):
    python3 predecirComentarioBert.py
