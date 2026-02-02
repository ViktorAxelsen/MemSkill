import regex
import string
import numpy as np
import os
import math
import copy
from collections import defaultdict
from json_repair import repair_json
import json
from collections import Counter
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from tqdm import tqdm


from llm_utils import get_llm_response


def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute the token-level F1 score between `prediction` and `ground_truth`.

    Steps:
        1. Return 0.0 immediately if prediction indicates an API request error.
        2. Normalize both strings (lowercase, remove punctuation, trim spaces, etc.).
        3. Tokenize and apply stemming (Porter Stemmer) to improve matching robustness.
        4. Count overlapping tokens using Counter intersection.
        5. If no overlap exists, return 0.
        6. Otherwise compute:
            precision = overlap_count / number_of_prediction_tokens
            recall    = overlap_count / number_of_ground_truth_tokens
            F1        = 2 * precision * recall / (precision + recall)
    """
    if prediction.lower() == "API Request Error".lower():
        return 0.0
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def f1_max(prediction, ground_truth):
    """
    Compute the multi-answer F1 score when both prediction and ground_truth
    may contain multiple comma-separated answers.

    For example:
        prediction     = "A, B, C"
        ground_truth   = "B, D"

    For each ground-truth answer `gt`:
        - Compute F1(pred_i, gt) for all predicted answers pred_i
        - Take the maximum F1 (best match)

    Final score:
        - Average the best-match F1 scores across all ground-truth entries

    In summary:
        score = mean_over_gt( max_over_pred( F1(pred, gt) ) )
    """
    if prediction.lower() == "API Request Error".lower():
        return 0.0
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]

    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def parse_judge_response(response):
    if response is None:
        print("Judge score parse failed. Response is None.")
        return 0.0
    if not isinstance(response, str):
        response = str(response)
    response = response.strip()
    if not response:
        print("Judge score parse failed. Response is empty.")
        return 0.0
    try:
        repaired = repair_json(response)
        return float(json.loads(repaired)["score"])
    except Exception as e:
        print(f"Judge score parse failed. Error: {e}. Raw response:", response)
        return 0.0


def llm_judge(task_args, args):
    judge_args = copy.deepcopy(args)
    judge_args.max_new_tokens = 1024
    judge_args.model = getattr(judge_args, "llm_judge_model", "openai/gpt-oss-120b")
    judge_args.temperature = 0.0
    judge_args.api = True

    new_task_args = []
    for i in task_args:
        new_task_args.append((i[0], i[1], judge_args))

    ret = get_llm_response(args=judge_args, task_args=new_task_args)
    score_list = []
    for _, response, _, _ in ret:
        score_list.append(parse_judge_response(response))

    return score_list


if __name__ == '__main__':
    pass

