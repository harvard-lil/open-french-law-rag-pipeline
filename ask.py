import os
import csv
import datetime

import click
import chromadb
import litellm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from const import (
    DATABASE_PATH,
    OUTPUT_PATH,
    VECTOR_SEARCH_COLLECTION_NAME,
    VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL,
    VECTOR_SEARCH_NORMALIZE_EMBEDDINGS,
    VECTOR_SEARCH_QUERY_PREFIX,
    VECTOR_SEARCH_MAX_RESULTS,
    LITELLM_OLLAMA_MODEL,
    LITELLM_OPENAI_MODEL,
    LITELLM_MODEL_TEMPERATURE,
    QUESTIONS,
    PROMPT_NO_CONTEXT_EN,
    PROMPT_WITH_CONTEXT_EN,
    PROMPT_NO_CONTEXT_FR,
    PROMPT_WITH_CONTEXT_FR,
)

load_dotenv()

litellm.telemetry = False


SETUPS = [
    {
        "group": "a_en",
        "model": LITELLM_OLLAMA_MODEL,
        "prompt": PROMPT_NO_CONTEXT_EN,
        "has_context": False,
        "lang": "en",
    },
    {
        "group": "b_en",
        "model": LITELLM_OLLAMA_MODEL,
        "prompt": PROMPT_WITH_CONTEXT_EN,
        "has_context": True,
        "lang": "en",
    },
    {
        "group": "c_en",
        "model": LITELLM_OPENAI_MODEL,
        "prompt": PROMPT_NO_CONTEXT_EN,
        "has_context": False,
        "lang": "en",
    },
    {
        "group": "d_en",
        "model": LITELLM_OPENAI_MODEL,
        "prompt": PROMPT_WITH_CONTEXT_EN,
        "has_context": True,
        "lang": "en",
    },
    {
        "group": "a_fr",
        "model": LITELLM_OLLAMA_MODEL,
        "prompt": PROMPT_NO_CONTEXT_FR,
        "has_context": False,
        "lang": "fr",
    },
    {
        "group": "b_fr",
        "model": LITELLM_OLLAMA_MODEL,
        "prompt": PROMPT_WITH_CONTEXT_FR,
        "has_context": True,
        "lang": "fr",
    },
    {
        "group": "c_fr",
        "model": LITELLM_OPENAI_MODEL,
        "prompt": PROMPT_NO_CONTEXT_FR,
        "has_context": False,
        "lang": "fr",
    },
    {
        "group": "d_fr",
        "model": LITELLM_OPENAI_MODEL,
        "prompt": PROMPT_WITH_CONTEXT_FR,
        "has_context": True,
        "lang": "fr",
    },
]
""" Experimental setups, associating group names with models, prompts and experiment conditions info. """


@click.command()
def ask() -> bool:
    """
    Runs the Open French Law RAG experiment.

    Asks the questions defined in const.QUESTIONS to:
    - An OpenAI model
    - An Open-source model via Ollama
    - (See const and README for details).

    Outputs results as CSV under OUTPUT_PATH: one file per model.

    Requires:
    - A valid vector store with french law embeddings (run "ingest.py" ahead of time)
    - A valid OpenAI API key, given via the OPENAI_API_KEY env var.
    - An Ollama server running at the address indicated at the OLLAMA_API_URL env var.

    This command comes with very little error handling on purpose:
    - It MUST crash and print trace if anything goes wrong.
    """
    ollama_api_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

    embedding_model = SentenceTransformer(VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL)

    chroma_client = chromadb.PersistentClient(
        path=DATABASE_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    chroma_collection = chroma_client.get_collection(name=VECTOR_SEARCH_COLLECTION_NAME)

    csv_filepath = os.path.join(OUTPUT_PATH, f"{int(datetime.datetime.now().timestamp())}.csv")

    output_format = {
        "group": "",
        "datetime": "",
        "model": "",
        "temperature": "",
        "question": "",
        "response": "",
    }

    #
    # Prepare output format.
    # `source_x_abc`` is based on VECTOR_SEARCH_MAX_RESULTS.
    #
    for i in range(0, VECTOR_SEARCH_MAX_RESULTS):
        output_format[f"source_{i+1}_identifier"] = ""
        output_format[f"source_{i+1}_url"] = ""
        output_format[f"source_{i+1}_distance"] = ""
        output_format[f"source_{i+1}_texte_nature"] = ""
        output_format[f"source_{i+1}_texte_titre"] = ""
        output_format[f"source_{i+1}_text_chunk"] = ""

    #
    # Init CSV file
    #
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(csv_filepath, "w+") as file:
        writer = csv.DictWriter(file, fieldnames=output_format.keys())
        writer.writeheader()

    # For each question:
    # - Generate embedding for question
    # - Pull relevant embeddings from vector store
    # - Prepare prompts and run them against LLMs
    # - Save
    for question in QUESTIONS:

        for lang in ["en", "fr"]:
            click.echo(f"Asking: {question[lang]}")

            question_text = question[lang]

            # Encode question as embedding and pull similar embeddings from vector store
            question_embedding = embedding_model.encode(
                f"{VECTOR_SEARCH_QUERY_PREFIX}{question_text}",
                normalize_embeddings=VECTOR_SEARCH_NORMALIZE_EMBEDDINGS,
            ).tolist()

            sources_raw = chroma_collection.query(
                query_embeddings=question_embedding,
                n_results=VECTOR_SEARCH_MAX_RESULTS,
            )

            # Prepare context for injection in prompt
            context = ""

            for source in sources_raw["metadatas"][0]:
                context += f"The following is an excerpt from \"{source['texte_titre']}\":\n"
                context += f"{source['text_chunk']}\n\n"

            # Ask question to each relevant group, compile metadata and save output to CSV
            for setup in SETUPS:
                if setup["lang"] != lang:
                    continue

                model = setup["model"]
                prompt = setup["prompt"]
                prompt = prompt.replace("{question}", question_text)
                prompt = prompt.replace("{context}", context)

                response = litellm.completion(
                    model=model,
                    messages=[{"content": prompt, "role": "user"}],
                    temperature=LITELLM_MODEL_TEMPERATURE,
                    api_base=ollama_api_url if model.startswith("ollama") else None,
                )

                output = dict(output_format)
                output["group"] = setup["group"]
                output["datetime"] = datetime.datetime.utcnow()
                output["model"] = model
                output["temperature"] = LITELLM_MODEL_TEMPERATURE
                output["question"] = question_text
                output["response"] = response["choices"][0]["message"]["content"]

                for i in range(0, len(sources_raw["metadatas"][0])):
                    # Only export context if it was used
                    if setup["has_context"] is False:
                        break

                    source = sources_raw["metadatas"][0][i]
                    distance = sources_raw["distances"][0][i]
                    legi_url = "https://www.legifrance.gouv.fr/loda/id/"

                    output[f"source_{i+1}_identifier"] = source["article_identifier"]
                    output[f"source_{i+1}_url"] = f"{legi_url}{source['article_identifier']}"
                    output[f"source_{i+1}_distance"] = distance
                    output[f"source_{i+1}_texte_nature"] = source["texte_nature"]
                    output[f"source_{i+1}_texte_titre"] = source["texte_titre"]
                    output[f"source_{i+1}_text_chunk"] = source["text_chunk"]

                with open(csv_filepath, "a") as file:
                    writer = csv.DictWriter(file, fieldnames=output.keys())
                    writer.writerow(output)


if __name__ == "__main__":
    ask()
