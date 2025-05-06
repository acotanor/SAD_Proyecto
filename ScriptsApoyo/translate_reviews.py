import pandas as pd
import ast
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
template = """You are a professional translator. The response need to be just 1 option, and only that nothing else. Translate the following text into informal English, only if the original text is not in English. If the original text is already in English, return it as is, just change ’ for ' and never use ’ use ' instead
Text: {text}
Translation:"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model, temperature=0.7)

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

    print(f"Processing column: {column}...")
    translated_reviews = []

    for i, review in enumerate(df[column]):
        if pd.isna(review):
            translated_reviews.append("")
            print(f"Review {i+1} is empty, skipping.")
            continue

        try:
            # Intentar interpretar listas o diccionarios, si aplica
            try:
                parsed_review = ast.literal_eval(review) if review.startswith(('{', '[')) else review
            except:
                parsed_review = review

            # Determinar qué partes traducir
            comments_to_translate = []

            if isinstance(parsed_review, dict):
                comments_to_translate = [str(v) for v in parsed_review.values() if isinstance(v, str)]

            elif isinstance(parsed_review, list):
                comments_to_translate = [
                    str(item.get('comments', '')) 
                    for item in parsed_review 
                    if isinstance(item, dict) and 'comments' in item
                ]

            else:
                comments_to_translate = [str(parsed_review)]

            # Traducir cada fragmento
            translated_parts = []
            for idx, comment in enumerate(comments_to_translate):
                if not comment:
                    continue
                try:
                    input_prompt = prompt.format(text=comment)
                    translated = model(input_prompt)
                    translated_parts.append(translated)
                    print(f"Translated part {idx+1} in review {i+1} (column '{column}'): {translated}")
                except Exception as e:
                    print(f"Error translating part in review {i+1} (column '{column}'): {e}")
                    translated_parts.append(comment)

            final_translation = " ".join(translated_parts)
            translated_reviews.append(final_translation)

        except Exception as e:
            print(f"Error processing review {i+1} in column '{column}': {e}")
            translated_reviews.append(str(review))  # fallback

    # Validar longitud
    if len(translated_reviews) != len(df):
        raise ValueError(f"Length mismatch for column '{column}': {len(translated_reviews)} translations vs {len(df)} rows.")

    # Agregar columna traducida
    df[column] = translated_reviews


# Guardar el resultado
print(f"Saving translated reviews to {args.output_csv}...")
df.to_csv(args.output_csv, index=False, encoding='utf-8', quoting=1)
print("Translation completed!")
