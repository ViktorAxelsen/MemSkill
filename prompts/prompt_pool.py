
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""



LONGMEMEVAL_ANSWER_PROMPT = """
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.
\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nShort Answer:
"""


LLM_JUDGE_GENERAL_PROMPT = """
You are an expert judge evaluating the quality of an answer for a QA task.
Your goal is to determine whether the model's answer correctly and sufficiently
answers the given question.

Read the following information carefully:

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Your evaluation criteria:
1. Correctness:
   - Is the model answer factually consistent with ANY of the correct answers?
   - Does it avoid contradictions or introducing false information?

2. Relevance:
   - Does the answer address the question directly without unnecessary content?

3. Completeness:
   - Does the answer include all essential information needed to fully answer the question?
   - Partial answers are allowed but should receive lower scores.

Scoring Rules:
- Score = 1.0 if the answer is fully correct.
- Score = 0.5 if the answer is partially correct but incomplete or slightly inaccurate.
- Score = 0.0 if the answer is incorrect, irrelevant, or contradicts the ground truth.

Output Format (STRICT):
Return your output as a JSON dictionary with two fields:
{{
    "explanation": "<brief explanation of your reasoning>",
    "score": <0.0 | 0.5 | 1.0>
}}

Be concise and objective. Do not include anything outside the JSON.
"""



HOTPOTQA_ANSWER_PROMPT = """Based on the following context, answer the question. The question may require reasoning across multiple pieces of information.

{context}

Question: {question}

Instructions:
- Read the context carefully and identify relevant information
- If the answer can be found in the context, provide a short, precise answer
- Output your answer within <answer></answer> tags

<answer>your answer here</answer>"""



