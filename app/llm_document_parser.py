from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_elasticsearch import ElasticsearchStore

## for vector store
from elasticsearch import Elasticsearch
from config import CONFIG


def parse_document(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def parse_triplets(filepath):
    docs = parse_document(filepath)
    # result = []
    # for i in range(len(docs) - 2):
    #     concat_str = (
    #         docs[i].page_content
    #         + " "
    #         + docs[i + 1].page_content
    #         + " "
    #         + docs[i + 2].page_content
    #     )
    #     result.append(concat_str)
    return docs


def load_document_embedding(filepath, hf, index_name):
    url = f"http://{CONFIG.username}:{CONFIG.password}@{CONFIG.endpoint}:9200"
    es = Elasticsearch(
        [url], basic_auth=("elastic", CONFIG.password), http_compress=True
    )

    ## Parse the document if necessary
    if not es.indices.exists(index=index_name):
        print(f"\tThe index: {index_name} does not exist")
        print(">> 1. Chunk up the Source document")

        docs = parse_triplets(filepath)

        print(">> 2. Index the chunks into Elasticsearch")

        elastic_vector_search = ElasticsearchStore.from_documents(
            docs,
            embedding=hf,
            es_url=url,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=True,
            ),
        )
    else:
        print("\tLooks like the document is already loaded, let's move on")
        elastic_vector_search = ElasticsearchStore(
            embedding=hf,
            es_url=url,
            index_name=index_name,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=True,
            ),
        )
    return elastic_vector_search
