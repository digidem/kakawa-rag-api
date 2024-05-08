import logging

from llama_index.postprocessor.colbert_rerank import ColbertRerank


def initialize_colbert_reranker(top_k):
    import time

    start_time = time.time()
    logging.info("Initializing ColbertRerank with top_k={}".format(top_k))
    retrieval_strategy = "ColbertRerank"
    colbert_reranker = ColbertRerank(
        top_n=top_k,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )
    initialization_time = time.time() - start_time
    logging.info(f"ColbertRerank initialized in {initialization_time:.2f} seconds")
    return colbert_reranker, retrieval_strategy
