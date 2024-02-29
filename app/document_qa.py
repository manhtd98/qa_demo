from llm_model import setup_embeddings, make_the_llm
from llm_vectorstore import setup_vectordb
from llm_document_parser import load_document_embedding
from config import CONFIG

# # Huggingface embedding setup
hf = setup_embeddings()
db, url = setup_vectordb(hf, CONFIG.index_name)
llm_chain_informed = make_the_llm()
load_document_embedding(CONFIG.source_path, url, hf, CONFIG.index_name)


# ## how to ask a question
def ask_a_question(question):
    similar_docs = db.similarity_search(question)
    print(f"The most relevant passage: \n\t{similar_docs[0].page_content}")
    informed_context = similar_docs[0].page_content
    response = llm_chain_informed.run(context=informed_context, question=question)
    return response


if __name__ == "__main__":
    command = "User Question>> "
    response = ask_a_question(command)
    print(f"\n\n Answer: {response}\n")
