import pandas as pd

def eliminar_columnas(csv_path, columnas_a_eliminar, output_path):
    """
    Elimina las columnas especificadas de un archivo CSV y guarda el resultado en un nuevo archivo.

    :param csv_path: Ruta del archivo CSV original.
    :param columnas_a_eliminar: Lista de nombres de columnas a eliminar.
    :param output_path: Ruta donde se guardará el nuevo archivo CSV.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # Guardar el nuevo archivo CSV
    df.to_csv(output_path, index=False)
    print(f"Archivo guardado en: {output_path}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo CSV original
    csv_path = "airbnb_portugal.csv"
    
    # Columnas a eliminar
    columnas_a_eliminar = ["listing_url","summary","space","description","neighborhood_overview","notes","transit","access","interaction","house_rules","bed_type","minimum_nights","maximum_nights","cancellation_policy","accommodates","bedrooms","beds","bathrooms","amenities","extra_people","guests_included","images","host","availability","weekly_price","monthly_price","text_embeddings","image_embeddings"]
    
    # Ruta del archivo CSV de salida
    output_path = "airbnb_portugal_limpio.csv"
    
    # Llamar a la función
    eliminar_columnas(csv_path, columnas_a_eliminar, output_path)