# models/categorization_model.py
# Lógica para cargar y usar el modelo PEFT de categorización (mt5_base)

import torch
import os
# Importamos AutoModelForSeq2SeqLM y AutoTokenizer ya que tu script de evaluación usa Seq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# No necesitamos PeftModel ni PeftConfig si cargamos el modelo fusionado/completo
# from peft import PeftModel, PeftConfig
import re  # Para parsear la cadena de salida

# Define la ruta donde guardaste tu modelo PEFT entrenado para categorización
# Esta ruta debe contener los archivos guardados por tu entrenamiento (config.json, model.safetensors/bin, tokenizer files)
# Ejemplo: Si tu modelo está en my_ss_chatbot/models/mt5_categorization
CATEGORIZATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mt5_base")  # <--- ¡AJUSTA ESTA RUTA!

# Parámetros para la generación del modelo (de tu script de evaluación)
GENERATION_MAX_LENGTH = 64
GENERATION_NUM_BEAMS = 8
GENERATION_DO_SAMPLE = False

# Patrón Regex para extraer Categoría y Subcategoría de la cadena de salida
# Esperamos un formato como "Categoria: [algo], Subcategoria: [algo]"
# Este patrón captura lo que sigue a "Categoria: " y "Subcategoria: "
CATEGORY_REGEX = re.compile(r"Categoria: (.*?), Subcategoria: (.*)")

def load_categorization_model():
    """
    Carga el modelo de categorización (Seq2Seq) y su tokenizador
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
    print(f"\n--- Cargando modelo de categorización ---")
    print(f"Directorio del modelo: {CATEGORIZATION_MODEL_PATH}")

    # --- Lógica para seleccionar el dispositivo (CUDA, MPS o CPU) ---
    device = torch.device("cpu")  # Valor por defecto

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
        # Opcional: Limitar threads si MPS da problemas de paralelismo
        # torch.set_num_threads(1)
    else:
        print("Ni CUDA ni MPS están disponibles. Usando CPU.")
    print(f"Usando dispositivo: {device}")

    # --- Cargar Tokenizador y Modelo ---
    try:
        # 1. Cargar el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(CATEGORIZATION_MODEL_PATH)
        print("Tokenizador cargado correctamente.")

        # 2. Cargar el modelo para Sequence-to-Sequence LM
        # Esto carga el modelo base (MT5) y, si se guardó correctamente,
        # la información del adaptador PEFT.
        model = AutoModelForSeq2SeqLM.from_pretrained(CATEGORIZATION_MODEL_PATH)
        print("Modelo AutoModelForSeq2SeqLM cargado correctamente.")

        # Mover el modelo al dispositivo seleccionado (CPU/GPU)
        model.to(device)
        print(f"Modelo movido a: {device}")

        # Poner el modelo en modo evaluación (importante para inferencia)
        model.eval()
        print("Modelo puesto en modo evaluación.")

    except Exception as e:
        print(f"Error al cargar el modelo o el tokenizador desde '{CATEGORIZATION_MODEL_PATH}': {e}")
        print(
            "Asegúrate de que la ruta es correcta y contiene los archivos necesarios (config.json, model.safetensors/bin, tokenizer files).")
        raise  # Relanzar la excepción

    print("--- Modelo de categorización cargado exitosamente ---")
    return tokenizer, model, device


def predict_category(text: str, tokenizer, model, device) -> tuple[str, str]:
    """
    Realiza una predicción de categoría y subcategoría para un texto dado
    usando el modelo Seq2Seq cargado.

    Args:
        text (str): El texto de la pregunta a clasificar.
        tokenizer: El tokenizador cargado.
        model: El modelo cargado y en modo evaluación.
        device: El dispositivo (torch.device) donde se encuentra el modelo.

    Returns:
        tuple[str, str]: Una tupla (categoría, subcategoría).
                         Retorna ("Desconocida", "Desconocida") si la salida
                         del modelo no se ajusta al formato esperado.
    Raises:
        Exception: Si ocurre un error durante la inferencia.
    """
    print(f"\n--- Iniciando predicción para: '{text}' en dispositivo {device} ---") # Log de inicio

    try:
        # 1. Preparar la entrada (la pregunta)
        # Construir el prompt exactamente como se hizo durante el entrenamiento
        prompt = f"Clasifica la siguiente pregunta en una categoria y subcategoria:\n{text}"
        print(f"  Prompt de entrada: '{prompt}'") # Log del prompt

        inputs = tokenizer(
            prompt,  # Usar el prompt, no solo el texto
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Usar el mismo max_length que en el entrenamiento/evaluación
        )
        print(f"  Inputs tokenizados: {inputs.keys()}") # Log de los inputs tokenizados

        # 2. Mover los inputs al mismo dispositivo que el modelo (CPU/GPU)
        # Corregido: Iterar sobre los items del diccionario inputs
        inputs = {key: tensor.to(device) for key, tensor in inputs.items() if key in ['input_ids', 'attention_mask']}
        print(f"  Inputs movidos a {device}") # Log de inputs en dispositivo
        # Asegurarse de que input_ids y attention_mask están presentes
        if 'input_ids' not in inputs or 'attention_mask' not in inputs:
             raise ValueError("Los inputs tokenizados no contienen 'input_ids' o 'attention_mask'.")


        # 3. Realizar la inferencia (Generación)
        model.eval()  # Asegurar modo evaluación
        print("  Llamando a model.generate()...") # Log antes de generate
        with torch.no_grad():
            # --- Simplificar parámetros de generación para depurar ---
            # Mantengamos los parámetros originales de tu script de evaluación por ahora,
            # ya que el error anterior no parecía estar relacionado con ellos.
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=GENERATION_MAX_LENGTH,
                num_beams=GENERATION_NUM_BEAMS, # Usar num_beams de config
                do_sample=GENERATION_DO_SAMPLE, # Usar do_sample de config
                # Añadir otros parámetros de generación si se usaron en evaluación
            )
            # -------------------------------------------------------
        print("  model.generate() completado.") # Log después de generate
        print(f"  Generated IDs shape: {generated_ids.shape}") # Log de la forma de los IDs generados

        # 4. Decodificar la salida generada a texto
        print("  Llamando a tokenizer.batch_decode()...") # Log antes de decode
        predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predicted_text = predicted_text.strip()
        print(f"  Predicted Text: '{predicted_text}'") # Log del texto predicho

        # 5. Parsear la cadena de salida para extraer Categoría y Subcategoría
        print(f"  Parseando texto: '{predicted_text}' con regex...") # Log antes de regex
        match = CATEGORY_REGEX.match(predicted_text)
        print(f"  Regex match result: {match}") # Log del resultado del match

        if match:
            category = match.group(1).strip()
            subcategory = match.group(2).strip()
            print(f"  Parsed: Category='{category}', Subcategory='{subcategory}'") # Log del parseo exitoso
            return category, subcategory
        else:
            print(f"Advertencia: La salida del modelo no se ajusta al formato esperado: '{predicted_text}'")
            return "Desconocida", "Desconocida"

    except Exception as e:
        print(f"Ocurrió un error durante la inferencia de categorización: {e}")
        raise  # Relanzar la excepción


# --- Ejemplo de Uso (para pruebas locales del módulo) ---
# Puedes ejecutar este archivo directamente con `python -m models.categorization_model`
# desde la raíz del proyecto para probar solo la carga y predicción del modelo de categorización.
if __name__ == "__main__":
    # Asegúrate de que CATEGORIZATION_MODEL_PATH está configurado correctamente arriba.
    # Asegúrate de que CATEGORY_REGEX coincide con el formato de salida de tu modelo.

    try:
        # Cargar el modelo y tokenizador
        categorization_tokenizer, categorization_model_loaded, device_used = load_categorization_model()

        print("\n--- Probando predicción de categorización ---")

        # Define preguntas de ejemplo
        # Asegúrate de que estas preguntas corresponden a temas que tu modelo
        # fue entrenado para generar en el formato esperado.
        test_questions = [
            "¿Dónde debería dirigirme para solicitar el IMV sin contratiempos?",
            "¿Es posible tramitar el IMV por internet o debo acudir en persona?",
            "¿Cuándo me puedo jubilar?",
            "No sé cómo va lo del Cl@ve, ¿me lo explican como para mi abuela?",  # Puede que no mapee si no entrenaste con esto
            "No entiendo las razones del rechazo de mi presentación"  # Probablemente no mapee
        ]

        # Clasificar cada pregunta de prueba
        for i, question in enumerate(test_questions):
            print(f"\nPregunta {i + 1}: '{question}'")
            try:
                category_result, subcategory_result = predict_category(question, categorization_tokenizer,
                categorization_model_loaded, device_used)
                print(f"Resultado: Categoría/Subcategoría: **{category_result} / {subcategory_result}**")
            except Exception as e:
                print(f"Error durante la predicción: {e}")

        print("\n--- Prueba de categorización finalizada ---")

    except Exception as e:
        print(f"\nError durante la inicialización o prueba del módulo: {e}")
        print("Por favor, verifica CATEGORIZATION_MODEL_PATH y el formato de salida esperado del modelo.")
# Lógica para cargar y usar el modelo PEFT de categorización (mt5_base)
