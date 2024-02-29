from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

## for vector store
from langchain.vectorstores import ElasticVectorSearch
from elasticsearch import Elasticsearch


def parse_document(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def parse_triplets(filepath):
    docs = parse_document(filepath)
    result = []
    for i in range(len(docs) - 2):
        concat_str = (
            docs[i].page_content
            + " "
            + docs[i + 1].page_content
            + " "
            + docs[i + 2].page_content
        )
        result.append(concat_str)
    return result


def load_document_embedding(filepath, url, hf, index_name):
    es = Elasticsearch([url])

    ## Parse the document if necessary
    if not es.indices.exists(index=index_name):
        print(f"\tThe index: {index_name} does not exist")
        print(">> 1. Chunk up the Source document")

        docs = parse_triplets(filepath)

        print(">> 2. Index the chunks into Elasticsearch")

        elastic_vector_search = ElasticVectorSearch.from_documents(
            docs,
            embedding=hf,
            elasticsearch_url=url,
            index_name=index_name,
            # ssl_verify = ssl_verify
        )
    else:
        print("\tLooks like the document is already loaded, let's move on")
