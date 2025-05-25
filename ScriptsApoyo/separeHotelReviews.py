import pandas as pd
import ast
import csv

# Ruta de entrada y salida
input_csv = 'csv_unido.csv'
output_csv = 'output_reviews.csv'


df = pd.read_csv(input_csv)

rows = []

for _, row in df.iterrows():
    pais = row['pais']
    hotel_id = row['_id']

    try:
        reviews = ast.literal_eval(row['reviews'])  # Convertir el string a lista de dicts
    except (ValueError, SyntaxError):
        reviews = []

    for review in reviews:
        review_date = review.get('date', '')
        comments = review.get('comments', '')
        rows.append([pais, hotel_id, review_date, comments])

output_df = pd.DataFrame(rows, columns=['pais', 'hotel_id', 'fecha_review', 'review'])
output_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"Archivo generado: {output_csv}")
