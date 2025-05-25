import pandas as pd

# Cargar los dos CSV
csv1 = pd.read_csv("spain_limpio_1.csv")
csv2 = pd.read_csv("portugal_limpio_1.csv")

# Concatenar verticalmente
df_unido = pd.concat([csv1, csv2], ignore_index=True)

# Guardar el resultado
df_unido.to_csv("csv_unido.csv", index=False)
