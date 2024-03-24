from llama_index.core import PromptTemplate


def update_prompt(query_engine):
    qa_prompt_tmpl_str = (
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
        "3. If the context is not relevant to the query, respond 'you're unaware'.\n"
        "user: Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {query_str}\n"
        "Answer: \n"
        "assistant: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
