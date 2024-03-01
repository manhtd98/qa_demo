from llm_model import setup_embeddings, make_the_llm
from llm_document_parser import load_document_embedding
from config import CONFIG

# # Huggingface embedding setup
hf = setup_embeddings()
llm_chain_informed = make_the_llm()
elastic_vector_search = load_document_embedding(
    CONFIG.source_path, hf, CONFIG.index_name
)


# ## how to ask a question
def ask_a_question(question):
    similar_docs = elastic_vector_search.similarity_search(question)
    print(f"The most relevant passage: \n\t{similar_docs[0].page_content}")
    informed_context = similar_docs[0].page_content
    response = llm_chain_informed.run(context=informed_context, question=question)
    return response


if __name__ == "__main__":
    command = "User Question>> What's twenty-seven times the most borrowed?"
    response = ask_a_question(command)
    print(f"\n\n Answer: {response}\n")
