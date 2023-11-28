# Open French Law RAG Pipeline

CLI utility for running the **Open French Law RAG experiment**.

- 📰 Blog post: https://lil.law.harvard.edu/blog/2025/01/21/open-french-law-rag/
- 📖 Case study: https://lil.law.harvard.edu/open-french-law-rag

**This Retrieval Augmented Generation pipeline:**
- Ingests the [COLD French Law Dataset](https://huggingface.co/datasets/harvard-lil/cold-french-law) into a vector store.
  - Only French content is ingested. English translations present in the dataset are not part of this experiment.
- Uses the resulting vector store and a combination of text generation models to answer a series of questions.
  - Questions are asked both in English and French.
  - Questions are asked both with _and_ without context retrieved from the vector store
  - Questions are asked against both an OpenAI model and open-source model, which is run via [Ollama](https://ollama.ai)
- Outputs raw results to CSV

---

## Usage
This pipeline requires [Python 3.11+](https://www.python.org/) and [Python Poetry](https://python-poetry.org/).

Pulling and pushing data from [HuggingFace](https://huggingface.co/) may require the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and [valid authentication](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login).

### 1. Clone this repository
```bash
git clone https://github.com/harvard-lil/open-french-law-chabot.git
```

### 2. Install dependencies
```bash
poetry install
```

### 3. Configure the application
Copy and edit `.env.example` as `.env` in order to provide the pipeline credentials to the OpenAI API and Ollama.

```bash
cp .env.example .env
```

The pipeline's configuration can be further edited via [/const/__init__.py](/const/__init__.py).

### 3. Run the "ingest" script
This script generates a vector store out of the content from the [COLD French Law Dataset](https://huggingface.co/datasets/harvard-lil/cold-french-law).

```bash
# See: ingest.py --help for a list of available options
poetry run python ingest.py
```

See output under `/database`.

### 4. Run the "ask" script
This scripts runs the full list of questions through the pipeline and writes the output to CSV. 

```bash
# See: ask.py --help for a list of available options
poetry run python ask.py
```

See output under `/output/*.csv`.

---

## Output groups

The experiment's output is organized in groups: 

| Group name | Text gen. Model | RAG | Language |
| --- | --- | --- | --- |
| `a_en` | LLama2 70B | NO | EN |
| `a_fr` | LLama2 70B | NO | FR |
| `b_en` | LLama2 70B | YES | EN |
| `b_fr` | LLama2 70B | YES | FR |
| `c_en` | GPT-4 | NO | EN |
| `c_fr` | GPT-4 | NO | FR |
| `c_en` | GPT-4 | YES | EN |
| `c_fr` | GPT-4| YES | FR |
