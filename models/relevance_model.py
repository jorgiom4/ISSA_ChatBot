# models/relevance_model.py
# Lógica para cargar y usar el modelo PEFT de relevancia (xlm-roberta-base)

import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Ya no necesitamos PeftModel ni PeftConfig si cargamos el modelo fusionado/completo
# from peft import PeftModel, PeftConfig

# Define la ruta donde guardaste tu modelo PEFT entrenado para relevancia
# Esta ruta debe contener los archivos guardados por tu entrenamiento (config.json, model.safetensors/bin, tokenizer files)
# Ejemplo: Si tu modelo está en my_ss_chatbot/models/xlm_roberta
RELEVANCE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "xlm_roberta") # <--- AJUSTA ESTA RUTA si es diferente a 'xlm_roberta'

# Define el ID de la etiqueta que corresponde a "relevante" (SS)
# Esto es CRUCIAL y debe coincidir con cómo etiquetaste tus datos de entrenamiento
# y el mapeo id2label de tu modelo (generalmente en config.json).
# Si entrenaste con 2 clases (0: No SS, 1: SS), entonces RELEVANT_LABEL_ID = 1
# Si entrenaste con 2 clases (0: SS, 1: No SS), entonces RELEVANT_LABEL_ID = 0
# Verifica el mapeo id2label en el archivo config.json de tu modelo.
RELEVANT_LABEL_ID = 1 # <--- ¡AJUSTA ESTO SI ES NECESARIO! (Asumiendo 1 es 'SS')
RELEVANT_LABEL_TEXT = "SS" # <--- ¡AJUSTA ESTO SI ES NECESARIO! (Texto asociado al ID relevante)


def load_relevance_model():
    """
    Carga el modelo de clasificación de relevancia y su tokenizador
    desde el directorio especificado.
    Selecciona automáticamente el dispositivo (CUDA, MPS o CPU).

    Returns:
        tuple: (tokenizer, model, device)
               - tokenizer: El tokenizador cargado.
               - model: El modelo cargado y en modo evaluación.
               - device: El dispositivo (torch.device) donde se cargó el modelo.
    Raises:
        Exception: Si ocurre un error durante la carga.
    """
    print(f"\n--- Cargando modelo de relevancia ---")
    print(f"Directorio del modelo: {RELEVANCE_MODEL_PATH}")

    # --- Lógica para seleccionar el dispositivo (CUDA, MPS o CPU) ---
    device = torch.device("cpu") # Valor por defecto

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA está disponible. Usando GPU (NVIDIA).")
    elif torch.backends.mps.is_available():
        try:
            # MPS requiere PyTorch >= 1.12
            device = torch.device("mps")
            print("MPS está disponible. Usando GPU (Apple Silicon).")
        except Exception as e:
            print(f"MPS disponible pero falló al crear el dispositivo: {e}. Cayendo a CPU.")
            device = torch.device("cpu")
    else:
        print("Ni CUDA ni MPS están disponibles. Usando CPU.")
    print(f"Usando dispositivo: {device}")

    # --- Cargar Tokenizador y Modelo ---
    try:
        # 1. Cargar el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(RELEVANCE_MODEL_PATH)
        print("Tokenizador cargado correctamente.")

        # 2. Cargar el modelo para Clasificación de Secuencia
        # Esto carga el modelo base MÁS la cabeza de clasificación binaria
        # y, si se guardó correctamente, la información del adaptador PEFT.
        model = AutoModelForSequenceClassification.from_pretrained(RELEVANCE_MODEL_PATH)
        print("Modelo AutoModelForSequenceClassification cargado correctamente.")

        # Mover el modelo al dispositivo seleccionado (CPU/GPU)
        model.to(device)
        print(f"Modelo movido a: {device}")

        # Poner el modelo en modo evaluación (importante para inferencia)
        model.eval()
        print("Modelo puesto en modo evaluación.")

        # Opcional: Verificar el mapeo id2label si existe en la configuración del modelo
        if hasattr(model.config, 'id2label'):
             print(f"Mapeo ID a Etiqueta del modelo: {model.config.id2label}")
             # Puedes añadir una verificación aquí para asegurar que RELEVANT_LABEL_ID
             # coincide con el mapeo del modelo si quieres ser más estricto.
             # Nota: El mapeo en config.json puede ser {0: 'LABEL_0', 1: 'LABEL_1'} por defecto
             # si no se especificaron etiquetas durante el entrenamiento/guardado.
             # En ese caso, confía en tu conocimiento de qué ID corresponde a 'SS'.
             expected_label_from_config = model.config.id2label.get(RELEVANT_LABEL_ID)
             if expected_label_from_config and expected_label_from_config != RELEVANT_LABEL_TEXT:
                  print(f"ADVERTENCIA: El ID relevante configurado ({RELEVANT_LABEL_ID}) se mapea a '{expected_label_from_config}' en la configuración del modelo, no a '{RELEVANT_LABEL_TEXT}'. Asegúrate de que RELEVANT_LABEL_ID es correcto.")
        else:
             print("Advertencia: El modelo no tiene el mapeo 'id2label' en su configuración.")
             print(f"Asumiendo que el ID {RELEVANT_LABEL_ID} corresponde a '{RELEVANT_LABEL_TEXT}' según la configuración de este script.")


    except Exception as e:
        print(f"Error al cargar el modelo o el tokenizador desde '{RELEVANCE_MODEL_PATH}': {e}")
        print("Asegúrate de que la ruta es correcta y contiene los archivos necesarios (config.json, model.safetensors/bin, tokenizer files).")
        raise # Relanzar la excepción para que el llamador la capture

    print("--- Modelo de relevancia cargado exitosamente ---")
    return tokenizer, model, device


def predict_relevance(text: str, tokenizer, model, device) -> bool:
    """
    Realiza una predicción de relevancia para un texto dado usando el modelo cargado.

    Args:
        text (str): El texto de la pregunta a clasificar.
        tokenizer: El tokenizador cargado.
        model: El modelo cargado y en modo evaluación.
        device: El dispositivo (torch.device) donde se encuentra el modelo.

    Returns:
        bool: True si la pregunta es relevante ('SS'), False si no ('No SS').
    Raises:
        Exception: Si ocurre un error durante la inferencia.
    """
    # print(f"Realizando predicción para: '{text}' en dispositivo {device}") # Log detallado

    try:
        # 1. Preparar la entrada (la pregunta)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 2. Mover los inputs al mismo dispositivo que el modelo (CPU/GPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 3. Realizar la inferencia
        model.eval() # Asegurar modo evaluación
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Procesar las salidas para obtener la clasificación
        # Para clasificación, los outputs.logits contienen las puntuaciones para cada clase.
        logits = outputs.logits

        # Encontrar la clase con la puntuación más alta (el ID de la clase predicha)
        predicted_class_id = torch.argmax(logits, dim=-1).item()

        # 5. Determinar la relevancia basándose en el ID predicho
        # Comparamos el ID predicho con el ID que definimos como "relevante" (SS)
        is_relevant = predicted_class_id == RELEVANT_LABEL_ID

        # Opcional: Log detallado de la predicción
        # print(f"  Logits: {logits.tolist()}")
        # print(f"  Predicted ID: {predicted_class_id}")
        # if hasattr(model.config, 'id2label'):
        #     print(f"  Predicted Label: {model.config.id2label.get(predicted_class_id, 'Unknown')}")
        # print(f"  Is Relevant (SS): {is_relevant}")

        return is_relevant

    except Exception as e:
        print(f"Ocurrió un error durante la inferencia de relevancia: {e}")
        raise # Relanzar la excepción


# --- Ejemplo de Uso (para pruebas locales del módulo) ---
# Puedes ejecutar este archivo directamente con `python -m models.relevance_model`
# desde la raíz del proyecto para probar solo la carga y predicción del modelo de relevancia.
if __name__ == "__main__":
    # Asegúrate de que RELEVANCE_MODEL_PATH y RELEVANT_LABEL_ID/TEXT
    # están configurados correctamente arriba.

    try:
        # Cargar el modelo y tokenizador
        relevance_tokenizer, relevance_model_loaded, device_used = load_relevance_model()

        print("\n--- Probando predicción de relevancia ---")

        # Define preguntas de ejemplo
        test_questions = [
            "¿Como se que tengo derecho al IMV?", # Debería ser True (SS)
            "¿Las patatas son un fruto o una verdura?", # Debería ser False (No SS)
            "¿Puedo solicitar el subsidio de desempleo si he trabajado menos de un año?", # Debería ser True (SS)
            "¿Cual es la capital de Francia?", # Debería ser False (No SS)
            "¿Cómo afectan las bajas temperaturas a la presión de los neumáticos?" # Debería ser False (No SS)
        ]

        # Clasificar cada pregunta de prueba
        for i, question in enumerate(test_questions):
            print(f"\nPregunta {i+1}: '{question}'")
            try:
                is_relevant_result = predict_relevance(question, relevance_tokenizer, relevance_model_loaded, device_used)
                print(f"Resultado: ¿Es relevante (SS)? **{is_relevant_result}**")
            except Exception as e:
                print(f"Error durante la predicción: {e}")

        print("\n--- Prueba de relevancia finalizada ---")

    except Exception as e:
        print(f"\nError durante la inicialización o prueba del módulo: {e}")
        print("Por favor, verifica RELEVANCE_MODEL_PATH y asegúrate de que tus archivos de modelo son correctos.")