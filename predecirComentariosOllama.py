import requests

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
    return data["response"]

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
    prompt = f"""Give a rating from 1 to 9 for the following hotel review. Only answer with the number.
Example 1:
Review: "Lot of Traffic noise due to old windows toward the street. And the  guys where delayed  more than 1 hour we where waiting 
in front of the building and couldtn get in contact with the guys. Finally we came inside then the cleaning team has not been there so it was dirty used beeds beer cans used towels bed linen etc. so i had to wait another 2 hours before i could use the apartment . that was not nice after  a long day of traveling"
Rating: 2

Example 2:
Review: "It's a nice flat. The location is good even though you have to walk a bit to the metro and quite a bit to the historic center. I liked that there was a restaurant in the same building."
Rating: 5


Now, rate this review:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

def prediccion_fewShots(comment):
    prompt = f"""Give a rating from 1 to 9 for the following hotel review. Only answer with the number.
Example 1:
Review: "We had a spledid time in the old centre of Porto. The appartment is very well situated next to the old Ribeira square. It's perfect to have such an appartment to your disposal, you feel home, and have a place to relax between the exploration of this very nice city. We thank Ana & Gonçalo, and we hope the appartment is free when we go back next year. Porto is charming original"
Rating: 9

Example 2:
Review: "It's a nice flat. The location is good even though you have to walk a bit to the metro and quite a bit to the historic center. I liked that there was a restaurant in the same building."
Rating: 5

Example 3:
Review: "Lot of Traffic noise due to old windows toward the street. And the  guys where delayed  more than 1 hour we where waiting"
Rating: 2

Example 4:
Review:  "The apartment was as we expected and I recommend it to other travelers."
Rating: 5


Now, rate this review:
Review: "{comment}"
Rating:"""
    return consulta_ollama(prompt)

if __name__ == "__main__":
    comment = input("Enter the review to rate: ")
    print("\n--- 0-shot ---")
    print(prediccion_0shot(comment))
    print("\n--- 1-shots ---")
    print(prediccion_1shot(comment))
    print("\n--- 2-shots ---")
    print(prediccion_2shots(comment))
    print("\n--- Few-shots ---")
    print(prediccion_fewShots(comment))