import time
import openai
import threading
import tiktoken
from transformers import AutoTokenizer
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm



MAX_CONTEXT_LENGTH = {
    # ----- OpenAI GPT -----
    'gpt-4-turbo': 128000,
    'gpt-4': 4096,
    'gpt-4-32k': 320000,
    'gpt-3.5-turbo-16k': 16000,
    'gpt-3.5-turbo-12k': 12000,
    'gpt-3.5-turbo-8k': 8000,
    'gpt-3.5-turbo-4k': 4000,
    'gpt-3.5-turbo': 4096,
    'gpt-4o': 128000,
    'gpt-4o-2024-08-06': 128000,
    "gpt-4o-mini-2024-07-18": 128000,

    # ----- LLaMA / Meta -----
    "meta-llama/llama-3-70b-chat-hf": 8192,
    "meta-llama/llama-3.2-3b-instruct": 8192,
    "meta/llama-3.1-70b-instruct": 128000,
    "meta/llama-3.3-70b-instruct": 128000,

    # ----- Gemma -----
    "google/gemma-7b-it": 8192,

    # ----- Qwen -----
    "qwen/qwen1.5-72b-chat": 32768,
    "qwen/qwen2.5-3b-instruct": 32768,
    "qwen/qwen2.5-7b-instruct": 32768,
    "qwen/qwen2.5-0.5b-instruct": 32768,
    "qwen/qwen3-next-80b-a3b-instruct": 262144,
    "qwen/qwen3-4b-instruct-2507": 262144,

    # ----- Mixtral / Mistral -----
    "mistralai/mixtral-8x7B-instruct-v0.1": 32768,
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo": 32768,
    "mistralai/mixtral-8x22b-instruct-v0.1": 65536,
}




_client_cache = {}        
_key_index = 0            
_client_lock = threading.Lock()


def _get_client_round_robin(
    api_keys,
    base_url="",
    max_retries=2,
    timeout=60
):
    global _key_index, _client_cache

    if isinstance(api_keys, str):
        api_keys = [api_keys]

    if not api_keys:
        raise ValueError("API KEY EMPTY")

    with _client_lock:
        key = api_keys[_key_index % len(api_keys)]
        _key_index += 1

        client = _client_cache.get(key)
        if client is None:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=key,
                max_retries=max_retries,
                timeout=timeout
            )
            _client_cache[key] = client

    return client


def get_llm_response_via_api(prompt,
                             LLM_MODEL="",
                             base_url="",
                             api_key="",
                             MAX_TOKENS=512,
                             TAU=1.0,
                             TOP_P=1.0,
                             SEED=42,
                             MAX_TRIALS=3,
                             TIME_GAP=5,
                             response_format=None):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    if not api_key:
        raise ValueError("API KEY EMPTY")

    completion = None
    trials_left = MAX_TRIALS

    while trials_left:
        trials_left -= 1
        client = _get_client_round_robin(
            api_keys=api_key,
            base_url=base_url,
            max_retries=2,
            timeout=60
        )
        try:
            api_params = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TAU,
                "top_p": TOP_P,
                "seed": SEED,
                "max_tokens": MAX_TOKENS,
            }

            if response_format is not None:
                api_params["response_format"] = response_format

            completion = client.chat.completions.create(**api_params)
            break
        except Exception as e:
            print(client.api_key, e)
            if "request timed out" in str(e).strip().lower():
                break
            print(client.api_key, "Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))

    contents = completion.choices
    meta_info = completion.usage
    completion_tokens = meta_info.completion_tokens
    prompt_tokens = meta_info.prompt_tokens
    # total_tokens = meta_info.total_tokens
    # print(completion_tokens, prompt_tokens, total_tokens)
    if len(contents) == 1:
        return contents[0].message.content, prompt_tokens, completion_tokens
    else:
        return [c.message.content for c in contents], prompt_tokens, completion_tokens



def get_tokenizer(model_name):
    lower_name = model_name.lower()
    is_gpt = (
        "gpt" in lower_name or
        "openai" in lower_name or
        lower_name.startswith("o1") or
        lower_name.startswith("o3") or
        lower_name.startswith("gpt-")
    )

    if is_gpt:
        # ---- OpenAI tokenizer (tiktoken) ----
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        return encoding

    else:
        if model_name == "meta/llama-3.3-70b-instruct":
            model_name = "meta-llama/Llama-3.1-70B-Instruct"
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return hf_tokenizer


def request_task(data):
    q_id, query_text, args = data
    try:
        response_format = getattr(args, 'response_format', None)

        answer, prompt_tokens, completion_tokens = get_llm_response_via_api(
            prompt=query_text,
            MAX_TOKENS=args.max_new_tokens,
            LLM_MODEL=args.model,
            TAU=args.temperature,
            base_url=args.api_base,
            api_key=args.api_key,
            response_format=response_format
        )
        # print(answer)
        success = True
    except Exception as e:
        print(e)
        answer = "API Request Error"
        prompt_tokens = 0
        completion_tokens = 0
        success = False

    return q_id, answer, (prompt_tokens, completion_tokens), success


def get_llm_response(args, task_args):
    """
    q_id must be int, and is sorted by ascend order (q_id can be non-continuous)
    """
    ret = []
    if args.api:
        full_task_args = list(task_args)
        func_round = args.round
        while func_round > 0:
            func_round -= 1
            if len(ret) != 0:
                ret.sort(key=lambda x: x[0], reverse=False)
                task_args = [i for ind, i in enumerate(full_task_args) if not ret[ind][-1]]
                ret = [i for i in ret if i[-1]]

            with ThreadPool(args.batch_size) as p:
                for r in tqdm(p.imap_unordered(request_task, task_args), total=len(task_args), desc="Processing",
                              ncols=100):
                    ret.append(r)

            if sum([1 if not i[-1] else 0 for i in ret]) == 0:
                break
    else:
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": query[1]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for query in task_args
        ]
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_new_tokens
        )
        all_outputs = []
        output_token_counts = []
        for i in tqdm(range(0, len(prompt_texts), args.batch_size)):
            batch_prompts = prompt_texts[i:i + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for output in outputs:
                clean_text = output.outputs[0].text.strip()
                all_outputs.append(clean_text)
                output_tokens = len(tokenizer.encode(clean_text, add_special_tokens=False))
                output_token_counts.append(output_tokens)

        input_token_counts = [
            len(tokenizer.encode(prompt_text, add_special_tokens=False))
            for prompt_text in prompt_texts
        ]
        ret = []
        for element in zip([i[0] for i in task_args], all_outputs, input_token_counts, output_token_counts, [True] * len(all_outputs)):
            ret.append((element[0], element[1], (element[2], element[3]), element[-1]))

    ret.sort(key=lambda x: x[0], reverse=False)

    return ret



if __name__ == '__main__':
    pass
