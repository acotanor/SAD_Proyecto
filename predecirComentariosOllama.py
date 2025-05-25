import requests
import pandas as pd
import ast
import sys
from tqdm import tqdm  # Añade esta importación al principio

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"  # Cambia por el modelo que tengas cargado en Ollama

def consulta_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    data = response.json()
    if "response" not in data:
        print("Ollama API error or unexpected response:")
        print(data)
        raise ValueError("No 'response' key in Ollama API response.")
    return data["response"].strip()

def prediccion_0shot(comment):
    prompt = f"""Give a rating from 1 to 9 for the following hotel review. Only answer with the number:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

def prediccion_1shot(comment):
    prompt = f"""Give a rating from 1 to 9 for the following hotel review. Only answer with the number.
Example 1:
Review: "We had a spledid time in the old centre of Porto. The appartment is very well situated next to the old Ribeira square. It's perfect to have such an appartment to your disposal, you feel home, and have a place to relax between the exploration of this very nice city. We thank Ana & Gonçalo, and we hope the appartment is free when we go back next year. Porto is charming original"
Rating: 9

Now, rate this review:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

def prediccion_2shots(comment):
    prompt = f"""Give a rating from 1 to 10 for the following hotel review. Only answer with the number.
Example 1:
Review: "We had a spledid time in the old centre of Porto. The appartment is very well situated next to the old Ribeira square. It's perfect to have such an appartment to your disposal, you feel home, and have a place to relax between the exploration of this very nice city. We thank Ana & Gonçalo, and we hope the appartment is free when we go back next year. Porto is charming original"
Rating: 9

Example 5:
Review :" It was our first time using Airbnb' and it was really good. Thanks for everything!"
Rating: 8

Now, rate this review:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

def prediccion_fewShots(comment):
    prompt = f"""Give a rating from 1 to 10 for the following hotel review. Only answer with the number.
Example 1:
Review: "We had a spledid time in the old centre of Porto. The appartment is very well situated next to the old Ribeira square. It's perfect to have such an appartment to your disposal, you feel home, and have a place to relax between the exploration of this very nice city. We thank Ana & Gonçalo, and we hope the appartment is free when we go back next year. Porto is charming original"
Rating: 10

Example 2:
Review: "It's a nice flat. The location is good even though you have to walk a bit to the metro and quite a bit to the historic center. I liked that there was a restaurant in the same building."
Rating: 6

Example 3:
Review: "Great stay, would come back!"
Rating: 8

Example 4:
Review:  "The apartment was as we expected and I recommend it to other travelers."
Rating: 5

Example 5:
Review :" "The apartment was filthy, smelled terrible, and nothing worked. The bed was broken, there were bugs everywhere, and the host never responded to our messages. We left after the first night because it was impossible to stay there. Absolutely the worst experience I've ever had.""
Rating: 1

Now, rate this review:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

def extraer_comments_lista(reviews_str):
    try:
        reviews_list = ast.literal_eval(reviews_str)
        return [item['comments'] for item in reviews_list if 'comments' in item and item['comments'].strip()]
    except Exception:
        return []

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Modo batch: python predecirComentariosOllama.py input.csv output.csv
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        df = pd.read_csv(input_csv)
        resultados = []
        total = 0
        # Primero, calcula la lista de todos los comentarios (máximo 100)
        all_comments = []
        for _, row in df.iterrows():
            comments = extraer_comments_lista(row['reviews'])
            for comment in comments:
                all_comments.append(comment)
                if len(all_comments) >= 100:
                    break
            if len(all_comments) >= 100:
                break
        total = len(all_comments)
        with tqdm(total=total, desc="Procesando reviews") as pbar:
            for comment in all_comments:
                pred = prediccion_fewShots(comment)  # Cambia aquí la estrategia si quieres probar otra
                resultados.append({'review': comment, 'prediccion': pred})
                pbar.update(1)
        pd.DataFrame(resultados).to_csv(output_csv, index=False)
        print(f"Predicciones guardadas en {output_csv}")
    else:
        # Modo interactivo
        comment = input("Enter the review to rate: ")
        print("\n--- 0-shot ---")
        print(prediccion_0shot(comment))
        print("\n--- 1-shot ---")
        print(prediccion_1shot(comment))
        print("\n--- 2-shots ---")
        print(prediccion_2shots(comment))
        print("\n--- Few-shots ---")
        print(prediccion_fewShots(comment))