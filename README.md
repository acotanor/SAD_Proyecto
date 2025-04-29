# Requisitos de uso:
Para instalar los requirements hay que ejecutar los siguientes comandos:

1. python3 -m venv /ruta/del/entorno/.venv
2. source /ruta/del/entorno/.venv/bin/activate
3. python3 -m pip install --upgrade pip
4. pip install -r requirements.txt

# Llamada a la plantilla:
1. Obtener una descripción sobre los argumentos de llamada:
    python3 clasificador.py --help 
2. Entrenar un modelo con la configuración por defecto de plantillai (sin alterar config.json):
    python3 clasificador.py -m train -f "archivo.csv" -a {naive_bayes/kNN/decision_tree/random_forest} -p "columna a predecir"
3. Probar el modelo:
    python3 clasificador.py -m "test"


Para cambiar la configuración solo hay que modificar los atributos de config.json.
