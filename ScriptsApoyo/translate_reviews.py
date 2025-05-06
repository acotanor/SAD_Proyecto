import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

# Configuración de argumentos
parser = argparse.ArgumentParser(description='Translate Airbnb reviews to English using Ollama LLM')
parser.add_argument('--model', type=str, default='gemma2:2b', help='Ollama model name')
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save the translated CSV file')
parser.add_argument('--review_columns', type=str, required=True, help='Comma-separated list of column names to translate')
parser.add_argument('--lang', type=str, default='en', help='Target language for translation')
parser.add_argument('--sample', type=int, default=-1, help='Number of reviews to process (-1 for all)')
args = parser.parse_args()

# Configuración del modelo y prompt
template = """You are a professional translator. The response need to be just 1 option, and only that nothing eles. Translate the following text into informal English, only if the original text is not in English. If the original text is already in English, return it as is. Do not add any additional information or context:
Text: {text}
Translation:"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model, temperature=0.7)  # Ajusta la temperatura para controlar la creatividad

# Cargar el archivo CSV
print(f"Loading data from {args.input_csv}...")
df = pd.read_csv(args.input_csv)

# Parsear las columnas objetivo
columns_to_translate = args.review_columns.split(',')

# Limitar el número de reseñas a procesar si se especifica
if args.sample > 0:
    df = df.head(args.sample)

# Traducir las columnas especificadas
for column in columns_to_translate:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the input CSV.")
    
    print(f"Translating column: {column}...")
    translated_reviews = []
    
    for i, review in enumerate(df[column]):
        if pd.isna(review):  # Saltar reseñas vacías
            translated_reviews.append("")
            continue

        # Crear el prompt con la reseña
        input_prompt = prompt.format(text=review)
        
        # Obtener la traducción del modelo
        try:
            translation = model(input_prompt)
            translated_reviews.append(translation)
            print(f"Translated review {i + 1}/{len(df)} in column '{column}'")
            print(translated_reviews[-1])  # Imprimir la traducción
        except Exception as e:
            print(f"Error translating review {i + 1} in column '{column}': {e}")
            translated_reviews.append("")
    
    # Reemplazar la columna original con la columna traducida
    df[column] = translated_reviews

# Guardar el DataFrame con las columnas traducidas
print(f"Saving translated reviews to {args.output_csv}...")
df.to_csv(args.output_csv, index=False)
print("Translation completed!")