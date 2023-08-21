MULTI_QA_GPT35_PROMPT_TEMPLATE = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with a {k} different question and answer pairs that can be used to test a student's reading comprehension abilities.
When coming up with this question/answer pair, each pair must be respond in the following format:

{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}

So in your final answer you should response with a list of {k} pairs in this format:

```
[{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}},
 {{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}},
 {{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
    }}
]
```

Please come up with a list of {k} question/answer pairs, in the specified list of JSONS format, for the following text:
----------------
{text}
"""


MULTI_QA_GPT4_PROMPT_TEMPLATE = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with a {k} different question and answer pairs that can be used to test a student's reading comprehension abilities.
When coming up with this question/answer pair, each pair must be respond in the following format:

{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
Please come up with a list of {k} question/answer pairs, in the specified list of dict, for the following text:
----------------
{text}
"""


GRADE_DOCS_PROMPT_TEMPLATE = """
You are a grader trying to determine if a set of retrieved documents will help a student answer a question. \n

Here is the question: \n
{query}

Here are the documents retrieved to answer question: \n
{result}

Here is the correct answer to the question: \n
{answer}

Criteria:
  relevance: Do all of the documents contain information that will help the student arrive that the correct answer to the question?"

Your response should be as follows:

GRADE: (Correct or Incorrect, depending if all of the documents retrieved meet the criterion)
(line break)
JUSTIFICATION: (Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use three sentences maximum. Keep the answer as concise as possible.)
"""