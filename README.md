# Chatbot de Soporte para la Seguridad Social Española

Este proyecto implementa un chatbot conversacional basado en agentes de IA para responder preguntas, dudas y guiar en trámites relacionados con la Seguridad Social española.

La aplicación utiliza una arquitectura de agentes orquestada dentro de una aplicación Streamlit para la interfaz de usuario y la lógica principal. Incorpora modelos de Machine Learning entrenados con PEFT para la clasificación de preguntas y un sistema de Retrieval Augmented Generation (RAG) basado en PostgreSQL con la extensión pgvector para recuperar información relevante de una base de conocimiento.

## Objetivos Funcionales

- Recibir preguntas de los ciudadanos a través de un chat conversacional.
- Determinar la relevancia de la pregunta respecto a la Seguridad Social.
- Categorizar la pregunta (categoría y subcategoría).
- Recuperar información relevante de una base de conocimiento (RAG).
- Generar una respuesta coherente y útil basada en la información recuperada.

## Arquitectura del Proyecto (con Streamlit)

```
+-----------------+       +-----------------------------------+
|                 |       |                                   |
|  Usuario Final  | <---> |  Aplicación Streamlit             |
|  (Navegador)    |       |  (UI + Orquestación de Agentes)   |
|                 |       |                                   |
+-----------------+       +--------+-----------+-------------+
                                    |           |
                                    |           |
                              +-----v--+  +-----v-----+  +-----v-----+
                              | Modelo |  | Modelo    |  | Sistema   |
                              | Relev. |  | Categ.    |  |   RAG     |
                              | (PEFT) |  | (PEFT)    |  | (Postgres |
                              +--------+  +-----------+  | Embeddings)|
                                                         +-----------+
                                                               ^
                                                               |
                                                         +-----v-----+
                                                         | Base de   |
                                                         | Datos     |
                                                         | (Postgres)|
                                                         | (Historial|
                                                         | Conversación)|
                                                         +-----------+
```

## Estructura de Directorios y Archivos

```
my_ss_chatbot/
├── venv/                 # Entorno virtual (ignorar en Git)
├── app.py                # Script principal de la aplicación Streamlit (UI + Orquestación)
├── agents/               # Lógica de cada agente (funciones o clases Python)
│   ├── __init__.py       # Permite importar la carpeta como módulo
│   ├── relevance_agent.py  # Lógica para determinar si una pregunta es relevante
│   ├── categorization_agent.py # Lógica para categorizar una pregunta
│   ├── rag_agent.py        # Lógica para interactuar con el sistema RAG
│   └── response_agent.py   # Lógica para generar la respuesta final
├── models/               # Carga y uso de tus modelos PEFT
│   ├── __init__.py
│   ├── relevance_model.py  # Carga y uso del modelo de relevancia
│   └── categorization_model.py # Carga y uso del modelo de categorización
├── rag/                  # Lógica para interactuar con Postgres Embeddings
│   ├── __init__.py
│   ├── retriever.py        # Lógica para realizar búsquedas vectoriales
│   ├── ingest.py           # Script para cargar datos al RAG (proceso inicial)
│   └── config.py           # Configuración específica del RAG (conexión DB, etc.)
├── database/             # Configuración general de la Base de Datos
│   ├── __init__.py
│   ├── db.py               # Configuración de la conexión a Postgres
│   └── models.py           # Definición de modelos para el historial (si se usa DB)
├── config/               # Archivos de configuración
│   └── settings.py         # Configuración general (credenciales, paths, etc.)
├── data/                 # Directorio para almacenar documentos fuente para el RAG
├── notebooks/            # Directorio opcional para Jupyter notebooks de exploración/pruebas
├── requirements.txt      # Lista de dependencias de Python
└── README.md             # Descripción del proyecto y guía de setup
```

## Configuración y Ejecución

1.  **Clonar el repositorio** (si aplica).
2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    # En Linux/macOS:
    source venv/bin/activate
    # En Windows:
    .env\Scriptsctivate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    # Asegúrate de tener las dependencias correctas en requirements.txt (puedes usar 'pip freeze > requirements.txt' después de instalar manualmente)
    ```
4.  **Configurar la base de datos PostgreSQL con pgvector.** (Ver sección RAG y Base de Datos)
5.  **Cargar los datos en el sistema RAG** ejecutando el script de ingesta. (Ver sección RAG)
6.  **Configurar las credenciales y paths** en `config/settings.py`.
7.  **Ejecutar la aplicación Streamlit:**
    ```bash
    streamlit run app.py
    ```

## Desarrollo Paso a Paso

El desarrollo se está realizando de forma incremental, implementando y validando cada componente (carga de modelos, agentes, RAG) antes de integrarlo completamente en el flujo de orquestación en `app.py`.

## Secciones Pendientes (Implementación)

- Carga y uso de Modelos PEFT (`models/`)
- Implementación de la lógica de cada Agente (`agents/`)
- Configuración y uso del Sistema RAG con Postgres/pgvector (`rag/`, `database/`)
- Integración de un Modelo LLM generativo para el Agente de Respuesta.
- Orquestación final en `app.py`.

## Ficheros del modelo LLM y tokenizador Huggingface
Los ficheros necesarios para realizar la inferencia son:
- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

