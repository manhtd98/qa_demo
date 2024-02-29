import os

template_informed = """
SYSTEM: You are an intelligent assistant helping the users with their questions on knowledge base documents.
Question: {question}

Strictly Use the following pieces of context to answer the question at the end. Think step-by-step and then answer. You can improve the wordings or grammar.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "Cannot determine related insights from user's queries."
 - If the context is empty, just say "Cannot find information from user's queries."

=============
{context}
=============

Question: {question}
Helpful Answer:
"""


class CONFIG:
    OPTION_CUDA_USE_GPU = os.getenv("OPTION_CUDA_USE_GPU", "False") == "True"
    cache_dir = "./cache"
    index_name = "flan"
    model_id = "google/flan-t5-large"
    source_path = "./"
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    endpoint = os.getenv("ES_SERVER", "ERROR")
    username = os.getenv("ES_USERNAME", "ERROR")
    password = os.getenv("ES_PASSWORD", "ERROR")
    # ssl_verify = {
    #     "verify_certs": False,
    #     "basic_auth": (username, password),
    # }
