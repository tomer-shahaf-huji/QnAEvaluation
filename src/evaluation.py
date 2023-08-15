from typing import List
from prompts import GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT, \
    GRADE_ANSWER_PROMPT_OPENAI, GRADE_DOCS_PROMPT


def run_evaluation(qa_rag_chain, retriver, qna_GT, grade_prompt):
    predictions_list = []
    retrieved_docs = []
    latencies_list = []

    for qna in qna_GT:
        question, answer = qna["question"], qna["answer"]
        qa_rag_chain_answer = qa_rag_chain({"query": question})["result"]
        predictions_list.append({"question": question, "answer": answer, "result": qa_rag_chain_answer})
        retrieved_docs = retriver.get_relevant_documents(query=question)

    answers_grade = grade_model_answer(qna_GT, predictions_list, grade_prompt)
    retrieval_grade = grade_model_retrieval(qna_GT, retrieved_docs, grade_prompt)

    return answers_grade, retrieval_grade, latencies_list, predictions_list


def grade_model_answer(predicted_dataset: List, predictions: List, grade_answer_prompt: str) -> List:

    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        prompt=prompt
    )

    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs


def grade_model_retrieval(gt_dataset: List, predictions: List, grade_docs_prompt: str):
    prompt = GRADE_DOCS_PROMPT

    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        prompt=prompt
    )

    graded_outputs = eval_chain.evaluate(
        gt_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )
    return graded_outputs