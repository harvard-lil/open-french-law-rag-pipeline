import os
from shutil import rmtree

import click
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from datasets import load_dataset


from const import (
    DATABASE_PATH,
    VECTOR_SEARCH_COLLECTION_NAME,
    VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL,
    VECTOR_SEARCH_DISTANCE_FUNCTION,
    VECTOR_SEARCH_NORMALIZE_EMBEDDINGS,
    VECTOR_SEARCH_CHUNK_PREFIX,
)


@click.command()
@click.option(
    "--sentence-transformer-device",
    default="cpu",
    help="On what device should Sentence Transformer run?",
)
def ingest(sentence_transformer_device: str) -> bool:
    """
    Generates embeddings for the COLD French Law Dataset and stores them into a ChromaDB collection.

    May require Hugging Face authentication via `huggingface-cli login`.
    """
    #
    # Clear up existing vector store, if any
    #
    rmtree(DATABASE_PATH, ignore_errors=True)
    os.makedirs(DATABASE_PATH, exist_ok=True)

    #
    # Initialize vector store, embedding model and dataset access
    #
    chroma_client = chromadb.PersistentClient(
        path=DATABASE_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    chroma_collection = chroma_client.create_collection(
        name=VECTOR_SEARCH_COLLECTION_NAME,
        metadata={"hnsw:space": VECTOR_SEARCH_DISTANCE_FUNCTION},
    )

    embedding_model = SentenceTransformer(
        VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL,
        device=sentence_transformer_device,
    )

    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=VECTOR_SEARCH_SENTENCE_TRANSFORMER_MODEL,
        tokens_per_chunk=embedding_model[0].max_seq_length - 4,
        chunk_overlap=25,
    )  # Note: The text splitter adjusts its cut-off based on the models' max_seq_length

    dataset = load_dataset(
        "harvard-lil/cold-french-law",
        data_files="cold-french-law.csv",
        split="train",
    )

    dataset_total = len(dataset)
    embeddings_total = 0
    dataset_i = 0
    click.echo(f"{dataset_total} entries to ingest.")

    # For each entry in the dataset:
    # - Split text into chunks of X tokens
    # - Generate embeddings and associated metadata
    # - Add to vector store
    for entry in dataset:
        dataset_i += 1
        text = entry_to_text(entry)  # Generate text for current entry
        text_chunks = text_splitter.split_text(text)  # Split text into chunks

        # Add VECTOR_SEARCH_CHUNK_PREFIX to every chunk
        for i in range(0, len(text_chunks)):
            text_chunks[i] = VECTOR_SEARCH_CHUNK_PREFIX + text_chunks[i]

        # Status update
        status = f"{dataset_i}/{dataset_total} | "
        status += f"{entry['article_identifier']} was split into {len(text_chunks)} chunks."
        click.echo(status)

        # Generate embeddings and meta data for each chunk
        embeddings = embedding_model.encode(
            text_chunks,
            normalize_embeddings=VECTOR_SEARCH_NORMALIZE_EMBEDDINGS,
        )

        documents = []
        metadatas = []
        ids = []

        for i in range(0, len(text_chunks)):
            documents.append(entry["article_identifier"])
            ids.append(f"{entry['article_identifier']}-{i+1}")

            metadata = {
                "article_identifier": entry["article_identifier"],
                "texte_nature": entry["texte_nature"] if entry["texte_nature"] else "",
                "texte_titre": entry["texte_titre"] if entry["texte_titre"] else "",
                "texte_ministere": entry["texte_ministere"] if entry["texte_ministere"] else "",
                "text_chunk": text_chunks[i][len(VECTOR_SEARCH_CHUNK_PREFIX) :],  # noqa
            }

            metadatas.append(metadata)

        embeddings = embeddings.tolist()
        embeddings_total += len(embeddings)

        # Store embeddings and metadata
        chroma_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    click.echo(f"Done - {embeddings_total} embeddings for {dataset_total} documents.")
    return True


def entry_to_text(entry: dict) -> str:
    """
    Generates an "embeddable" text version of a `harvard-lil/cold-french-law` dataset record.
    """
    output = ""

    # Pick a "title"  based on texte_nature
    if entry["texte_nature"] == "CODE":
        output = f"Article {entry['article_num']} du {entry['texte_titre_court']}. "
    else:
        output = f"{entry['texte_titre']}. "

    # Remove line-breaks to increase N tokens per embedding
    text = entry["article_contenu_text"] if entry["article_contenu_text"] else ""
    output += text.replace("\n", " ")

    return output


if __name__ == "__main__":
    ingest()
