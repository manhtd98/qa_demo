import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

from config import template_informed, CONFIG
from langchain.embeddings import HuggingFaceEmbeddings


def setup_embeddings():
    print(f">> Prep. Huggingface embedding: {CONFIG.embedding_model} setup!")
    return HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)


def getFlanLarge():
    print(f">> Prep. Get {CONFIG.model_id} ready to go!")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)
    if CONFIG.OPTION_CUDA_USE_GPU:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CONFIG.model_id,
            cache_dir=CONFIG.cache_dir,
            load_in_8bit=True,
            device_map="auto",
        )
        model.cuda()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CONFIG.model_id, cache_dir=CONFIG.cache_dir
        )

    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def make_the_llm():
    local_llm = getFlanLarge()
    prompt_informed = PromptTemplate(
        template=template_informed, input_variables=["context", "question"]
    )

    return LLMChain(prompt=prompt_informed, llm=local_llm)
