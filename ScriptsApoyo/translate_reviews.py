import pandas as pd
import ast
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import argparse

# Argument configuration
parser = argparse.ArgumentParser(description='Translate Airbnb reviews to English using Ollama LLM')
mutex = parser.add_mutually_exclusive_group()
mutex.add_argument('--sample', type=int, default=-1, help='Number of reviews to process (use either --sample or --row_range)')
mutex.add_argument('--row_range', type=str, default='', help='Range of row indices (0-based) to process, e.g. 5-10')

parser.add_argument('--model', type=str, default='gemma2:2b', help='Ollama model name')
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save the translated CSV file')
parser.add_argument('--review_columns', type=str, required=True, help='Comma-separated list of column names to process')
parser.add_argument('--subcolumn', type=str, default=None, help='[Optional] Name of the subcolumn to translate in array-type reviews')
parser.add_argument('--lang', type=str, default='en', help='Target language for translation')
args = parser.parse_args()

# Parse row range
row_indices = set()
if args.row_range:
    try:
        start, end = map(int, args.row_range.split('-'))
        row_indices = set(range(start, end + 1))
    except ValueError:
        raise ValueError("Invalid --row_range format. Use 'start-end'")
elif args.sample > 0:
    row_indices = set(range(args.sample))

# Model configuration
template = (
    "You are a professional translator. Respond with only the translation. "
    "Translate to informal English if not already English. "
    "Use straight apostrophes (') instead of curly ones (’).\n"
    "Text: {text}\nTranslation:"
)
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model, temperature=0.7)

# Load CSV data
df = pd.read_csv(args.input_csv)
columns_to_process = [c.strip() for c in args.review_columns.split(',')]

def translate_text(text):
    """Translate individual text entries"""
    if pd.isna(text) or not text.strip():
        return text
    
    try:
        translated = model(prompt.format(text=text))
        return translated.replace('\n', ' ').replace('’', "'").strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def process_entry(entry):
    """Handle both array-type and simple string reviews"""
    # Try to parse as array first
    try:
        reviews = ast.literal_eval(entry)
        if not isinstance(reviews, list):
            return translate_text(entry)
        
        # Process array-type review
        processed = []
        for item in reviews:
            if isinstance(item, dict) and args.subcolumn and args.subcolumn in item:
                # Translate specified subcolumn
                item[args.subcolumn] = translate_text(item[args.subcolumn])
            processed.append(item)
        return str(processed)
    
    except (SyntaxError, ValueError):
        # Process as simple string
        return translate_text(entry)

# Process specified columns
for col in columns_to_process:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV")
    
    print(f"Processing column: {col}")
    df[col] = [
        process_entry(entry) if idx in row_indices or not row_indices else entry
        for idx, entry in enumerate(df[col])
    ]

# Save results
df.to_csv(args.output_csv, index=False, encoding='utf-8', quoting=1)
print("Translation completed successfully!")