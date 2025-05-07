import pandas as pd
import ast
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

# Configuración de argumentos
group = argparse.ArgumentParser(description='Translate Airbnb reviews to English using Ollama LLM')
mutex = group.add_mutually_exclusive_group()
mutex.add_argument('--sample', type=int, default=-1,
                   help='Number of reviews to process (use either --sample or --row_range)')
mutex.add_argument('--row_range', type=str, default='',
                   help='Range of row indices (0-based) to process, e.g. 5-10')
group.add_argument('--model', type=str, default='gemma2:2b',
                   help='Ollama model name')
group.add_argument('--input_csv', type=str, required=True,
                   help='Path to the input CSV file')
group.add_argument('--output_csv', type=str, required=True,
                   help='Path to save the translated CSV file')
group.add_argument('--review_columns', type=str, required=True,
                   help='Comma-separated list of column names to translate')
group.add_argument('--lang', type=str, default='en',
                   help='Target language for translation')
args = group.parse_args()

# Parsear rango de índices
if args.row_range:
    try:
        start_str, end_str = args.row_range.split('-')
        start, end = int(start_str), int(end_str)
        row_indices = set(range(start, end + 1))
    except ValueError:
        raise ValueError("--row_range must be in the form start-end, e.g. '0-5'")
elif args.sample > 0:
    row_indices = set(range(args.sample))
else:
    row_indices = set()  # vacío significa traducir todo

# Configuración del modelo y prompt
template = (
    "You are a professional translator. The response need to be just 1 option, and only that nothing else. "
    "Translate the following text into informal English, only if the original text is not in English. "
    "If the original text is already in English, return it as is, just change ’ for ' and never use ’ use ' instead\n"
    "Text: {text}\nTranslation:"
)
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model, temperature=0.7)

# Cargar el archivo CSV
print(f"Loading data from {args.input_csv}...")
df = pd.read_csv(args.input_csv)
columns_to_translate = [c.strip() for c in args.review_columns.split(',')]

# Función para decidir si procesar un índice
def should_translate(idx):
    if row_indices:
        return idx in row_indices
    return True

# Traducir las columnas especificadas
for column in columns_to_translate:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the input CSV.")

    print(f"Processing column: {column}...")
    translated_reviews = []

    for i, review in enumerate(df[column]):
        if not should_translate(i):
            translated_reviews.append(review if pd.notna(review) else "")
            print(f"Skipping row {i}.")
            continue

        if pd.isna(review):
            translated_reviews.append("")
            print(f"Review {i} is empty, skipping.")
            continue

        try:
            try:
                parsed_review = ast.literal_eval(review) if review.startswith(('{', '[')) else review
            except Exception:
                parsed_review = review

            if isinstance(parsed_review, dict):
                comments_to_translate = [str(v) for v in parsed_review.values() if isinstance(v, str)]
            elif isinstance(parsed_review, list):
                comments_to_translate = [str(item.get('comments', ''))
                                         for item in parsed_review
                                         if isinstance(item, dict) and 'comments' in item]
            else:
                comments_to_translate = [str(parsed_review)]

            translated_parts = []
            for idx_part, comment in enumerate(comments_to_translate):
                if not comment:
                    continue
                input_prompt = prompt.format(text=comment)
                try:
                    translated = model(input_prompt)
                    translated_parts.append(translated)
                    print(f"Translated part {idx_part+1} in row {i}: {translated}")
                except Exception as e:
                    print(f"Error translating part in row {i}: {e}")
                    translated_parts.append(comment)

            final_translation = " ".join(translated_parts)
            translated_reviews.append(final_translation)

        except Exception as e:
            print(f"Error processing review {i}: {e}")
            translated_reviews.append(str(review))

    if len(translated_reviews) != len(df[column]):
        raise ValueError(f"Length mismatch for column '{column}': {len(translated_reviews)} vs {len(df)} rows.")

    df[column] = translated_reviews

print(f"Saving translated reviews to {args.output_csv}...")
df.to_csv(args.output_csv, index=False, encoding='utf-8', quoting=1)
print("Translation completed!")