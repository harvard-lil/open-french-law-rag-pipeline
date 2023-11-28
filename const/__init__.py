import os

DATABASE_PATH = os.path.join(os.getcwd(), "database")
""" Path to the vector database folder. Absolute. """

OUTPUT_PATH = os.path.join(os.getcwd(), "output")
""" Path to the experiment output storage folder. Absolute. """

VECTOR_SEARCH_COLLECTION_NAME = "french-law-embeddings"
""" Name to be given to the ChromaDB collection. """

VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL = "intfloat/multilingual-e5-large"
"""
Sentence Transformer model to be used to generate embeddings.
More info: https://www.sbert.net/docs/pretrained_models.html
"""

VECTOR_SEARCH_DISTANCE_FUNCTION = "cosine"
"""
Distance function to be used by the vector store.
Can be "l2" (squared l2, euclidian), "ip" (inner product), "cosine" (cosine similarity).
More info: https://docs.trychroma.com/usage-guide#changing-the-distance-function
"""

VECTOR_SEARCH_NORMALIZE_EMBEDDINGS = True
"""
If `true`, sentence transformers will normalize the embeddings it generates. 
"""

VECTOR_SEARCH_CHUNK_PREFIX = "passage: "
"""
Certain embedding models expect text to be prepended with keywords based on context.
Example: intfloat/multilingual-e5-large expects "passage: " for text excerpts. 
"""

VECTOR_SEARCH_QUERY_PREFIX = "query: "
"""
Certain embedding models expect text to be prepended with keywords based on context.
Example: intfloat/multilingual-e5-large expects "query: " for search queries. 
"""

VECTOR_SEARCH_MAX_RESULTS = 4
""" Max results to be pulled from vector store. """

LITELLM_OLLAMA_MODEL = "ollama/llama2:70b-chat-fp16"
""" LiteLLM config. Name of the model to be used with the Ollama API """

LITELLM_OPENAI_MODEL = "gpt-4"
""" LiteLLM config. Name of the model to be used with the OpenAI API"""

LITELLM_MODEL_TEMPERATURE = 0.0
""" LiteLLM config. Temperature to be used across models. """

QUESTIONS = [
    {
        "en": "How long does it take for a vehicle left parked on the highway to be impounded?",
        "fr": "Au bout de combien de temps un véhicule laissé en stationnement sur le bord de d'une route peut-il être mis en fourrière?",
    },
    {
        "en": "Identify if an impact study is needed to develop a campsite.",
        "fr": "Identifie si une étude d'impact est nécessaire pour ouvrir un camping.",
    },
    {
        "en": "Explain whether any environmental authorization is needed to develop a rabbit farm.",
        "fr": "Explique si une autorisation environnementale est nécessaire pour ouvrir une ferme de lapins.",
    },
    {
        "en": "Who should compensate for damage to a maize field caused by wild boar?",
        "fr": "Qui a l'obligation de réparer les dommages causés à un champ de maïs par des sangliers sauvages?",
    },
    {
        "en": "Can a cow be considered as real estate?",
        "fr": "Une vache peut-elle être considérée comme un immeuble?",
    },
    {
        "en": "Is it legal for a fisherman to use an electric pulse trawler?",
        "fr": "Est-il légal d'utiliser un chalut électrique?",
    },
    {
        "en": "List the environmental principles in environmental law.",
        "fr": "Liste les principes du droit de l'environnement.",
    },
    {
        "en": "Identify and summarize the provisions for animal well-being.",
        "fr": "Identifie et résume le droit du bien être animal.",
    },
    {
        "en": "Identify and summarize the provisions for developing protected areas in environmental law.",
        "fr": "Identifie et résume le droit applicable à la création d'aires protégées en droit de l'environnement.",
    },
    {
        "en": "Is it legal to build a house within one kilometer of the coastline?",
        "fr": "Est-il légal de construire une maison à un kilomètre du rivage?",
    },
]
""" Questions to be asked to the chatbot. """

PROMPT_NO_CONTEXT_EN = """
You are a helpful and friendly legal assistant with expertise in French law.
You are here to support American users trying to understand French laws and regulations. Your objective is to answer their question by providing relevant information about French laws. Your explanation should be easy to understand while still being accurate and detailed.

Use your knowledge to answer the following QUESTION.

QUESTION: {question}

Helpful answer:
"""  # noqa
""" Prompt to be used in non-RAG context. Asks to respond in english. {context} placeholder needs to be replaced. """


PROMPT_WITH_CONTEXT_EN = """
Here is CONTEXT:
{context}
----------------
You are a helpful and friendly legal assistant with expertise in French law.
You are here to support American users trying to understand French laws and regulations. Your objective is to answer their question by providing relevant information about French laws. Your explanation should be easy to understand while still being accurate and detailed.

When possible, use the provided CONTEXT to answer the following QUESTION, but ignore CONTEXT if it is empty or not relevant.
When possible and relevant, use CONTEXT to cite specific french law codes and articles.

QUESTION: {question}

Helpful answer:
"""  # noqa
""" (Prompt to be used in RAG context. Asks to respond in english. {context} and {question} placeholders need to be replaced. """


PROMPT_NO_CONTEXT_FR = """
You are a helpful and friendly legal assistant with expertise in French law.
You are here to support American users trying to understand French laws and regulations. Your objective is to answer their question by providing relevant information about French laws. Your explanation should be easy to understand while still being accurate and detailed.

Use your knowledge to answer the following QUESTION.

QUESTION: {question}

Helpful answer (in french):
"""  # noqa
""" Prompt to be used in non-RAG context. Asks to respond in french. {context} placeholder needs to be replaced. """


PROMPT_WITH_CONTEXT_FR = """
Here is CONTEXT:
{context}
----------------
You are a helpful and friendly legal assistant with expertise in French law.
You are here to support American users trying to understand French laws and regulations. Your objective is to answer their question by providing relevant information about French laws. Your explanation should be easy to understand while still being accurate and detailed.

When possible, use the provided CONTEXT to answer the following QUESTION, but ignore CONTEXT if it is empty or not relevant.
When possible and relevant, use CONTEXT to cite specific french law codes and articles.

QUESTION: {question}

Helpful answer (in french):
"""  # noqa
""" Prompt to be used in RAG context. Asks to respond in french. {context} and {question} placeholders need to be replaced. """
