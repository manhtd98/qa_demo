from langchain.vectorstores import ElasticVectorSearch
from config import CONFIG


def setup_vectordb(hf, index_name):
    # Elasticsearch URL setup
    print(">> Prep. Elasticsearch config setup")

    url = f"https://{CONFIG.username}:{CONFIG.password}@{CONFIG.endpoint}:443"

    return (
        ElasticVectorSearch(embedding=hf, elasticsearch_url=url, index_name=index_name),
        url,
    )
