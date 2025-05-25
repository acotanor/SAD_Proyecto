import pandas as pd
import ast

# Cargar el CSV
df = pd.read_csv("airbnb_portugal_limpio_trans.csv")

# Procesar la columna 'review_scores'
def extraer_rating(review_scores):
    try:
        data = ast.literal_eval(review_scores)
        rating = data.get('review_scores_rating', None)
        return round(rating / 10, 1) if rating is not None else None
    except (ValueError, SyntaxError, TypeError):
        return None

df['review_score'] = df['review_scores'].apply(extraer_rating)

# Procesar la columna 'address'
def extraer_datos_address(address):
    try:
        data = ast.literal_eval(address)
        city = data.get('market', None)
        coordinates = data.get('location', {}).get('coordinates', [None, None])
        longitude, latitude = coordinates if len(coordinates) == 2 else (None, None)
        return pd.Series([city, latitude, longitude])
    except (ValueError, SyntaxError):
        return pd.Series([None, None, None])

df[['city', 'latitude', 'longitude']] = df['address'].apply(extraer_datos_address)

# Eliminar las columnas originales
df.drop(['review_scores', 'address'], axis=1, inplace=True)

# Guardar el nuevo CSV
df.to_csv("portugal_limpio_1.csv", index=False)
