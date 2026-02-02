from typing import List

import torch
import threading
import torch.nn.functional as F
from tqdm import tqdm




try:
    import faiss
except ImportError:
    print("FAISS INITIALIZATION FAILED")
    faiss = None

try:
    from langchain_community.retrievers.bm25 import BM25Retriever
    from langchain_community.retrievers.tfidf import TFIDFRetriever
    from langchain_core.documents import Document
except ImportError:
    print("LANGCHAIN INITIALIZATION FAILED")
    BM25Retriever = None
    TFIDFRetriever = None
    Document = None


# Global cache for retriever models to avoid repeated loading
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE_LOADING = {}


def _load_hf_model(model_cls, model_name: str, device: str):
    """Load HF model on a single device to avoid meta tensor dispatch issues."""
    def _load_with_kwargs(**kwargs):
        return model_cls.from_pretrained(model_name, **kwargs)

    def _has_meta_params(model):
        return any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers())

    def _load_no_map():
        try:
            return _load_with_kwargs(device_map=None, low_cpu_mem_usage=False)
        except TypeError:
            return _load_with_kwargs()

    # First try: normal load on CPU -> move to device.
    try:
        model = _load_no_map()
        model = model.to(device)
        if not _has_meta_params(model):
            model.eval()
            return model
    except RuntimeError as exc:
        if "meta tensor" not in str(exc).lower():
            raise

    # Fallback: let HF dispatch directly to device (skip .to()).
    try:
        model = _load_with_kwargs(device_map={"": device}, low_cpu_mem_usage=True)
        model.eval()
        return model
    except Exception:
        pass

    # Last resort: use to_empty + load_state_dict.
    model = _load_no_map()
    if _has_meta_params(model):
        model = model.to_empty(device=device)
        state_model = _load_no_map()
        model.load_state_dict(state_model.state_dict(), strict=False)
        del state_model
    else:
        model = model.to(device)
    model.eval()
    return model


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [B, L, D]
    mask: [B, L]
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)

    return sentence_embeddings


def init_context_model(retriever: str):
    """Initialize context encoder with global caching."""
    cache_key = f"{retriever}_context"
    while True:
        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                tokenizer, model = _MODEL_CACHE[cache_key]
                has_meta = any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers())
                uses_device_map = bool(getattr(model, "hf_device_map", None))
                if has_meta and not uses_device_map:
                    del _MODEL_CACHE[cache_key]
                    print("NO USING CACHE FOR RETRIEVER")
                else:
                    return tokenizer, model

            event = _MODEL_CACHE_LOADING.get(cache_key)
            if event is None:
                event = threading.Event()
                _MODEL_CACHE_LOADING[cache_key] = event
                is_loader = True
            else:
                is_loader = False

        if not is_loader:
            event.wait()
            continue
        break

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if retriever == 'dpr':
            from transformers import (
                DPRContextEncoder,
                DPRContextEncoderTokenizer,
            )
            context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            )
            context_model = _load_hf_model(
                DPRContextEncoder,
                "facebook/dpr-ctx_encoder-single-nq-base",
                device
            )
        elif retriever == 'contriever':
            from transformers import AutoTokenizer, AutoModel
            context_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            context_model = _load_hf_model(AutoModel, 'facebook/contriever', device)
        elif retriever == 'dragon':
            from transformers import AutoTokenizer, AutoModel
            context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-context-encoder')
            context_model = _load_hf_model(AutoModel, 'facebook/dragon-plus-context-encoder', device)
        else:
            raise ValueError(f"Unknown retriever type: {retriever}")
    
    except Exception:
        with _MODEL_CACHE_LOCK:
            event = _MODEL_CACHE_LOADING.pop(cache_key, None)
            if event is not None:
                event.set()
        raise
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (context_tokenizer, context_model)
        event = _MODEL_CACHE_LOADING.pop(cache_key, None)
        if event is not None:
            event.set()
    return context_tokenizer, context_model


def init_query_model(retriever: str):
    """Initialize query encoder with global caching."""
    cache_key = f"{retriever}_query"
    while True:
        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                tokenizer, model = _MODEL_CACHE[cache_key]
                has_meta = any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers())
                uses_device_map = bool(getattr(model, "hf_device_map", None))
                if has_meta and not uses_device_map:
                    del _MODEL_CACHE[cache_key]
                else:
                    return tokenizer, model

            event = _MODEL_CACHE_LOADING.get(cache_key)
            if event is None:
                event = threading.Event()
                _MODEL_CACHE_LOADING[cache_key] = event
                is_loader = True
            else:
                is_loader = False

        if not is_loader:
            event.wait()
            continue
        break

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if retriever == 'dpr':
            from transformers import (
                DPRQuestionEncoder,
                DPRQuestionEncoderTokenizer,
            )
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            )
            question_model = _load_hf_model(
                DPRQuestionEncoder,
                "facebook/dpr-question_encoder-single-nq-base",
                device
            )
        elif retriever == 'contriever':
            from transformers import AutoTokenizer, AutoModel
            question_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            question_model = _load_hf_model(AutoModel, 'facebook/contriever', device)
        elif retriever == 'dragon':
            from transformers import AutoTokenizer, AutoModel
            question_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
            question_model = _load_hf_model(AutoModel, 'facebook/dragon-plus-query-encoder', device)
        else:
            raise ValueError(f"Unknown retriever type: {retriever}")
    
    except Exception:
        with _MODEL_CACHE_LOCK:
            event = _MODEL_CACHE_LOADING.pop(cache_key, None)
            if event is not None:
                event.set()
        raise
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (question_tokenizer, question_model)
        event = _MODEL_CACHE_LOADING.pop(cache_key, None)
        if event is not None:
            event.set()
    return question_tokenizer, question_model


def get_embeddings(retriever: str,
                   inputs: List[str],
                   mode: str = 'context',
                   batch_size: int = 256) -> torch.Tensor:
    if mode == 'context':
        tokenizer, encoder = init_context_model(retriever)
    else:
        tokenizer, encoder = init_query_model(retriever)

    all_embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="GET EMBEDDINGS", leave=False):
            batch_texts = inputs[i:(i + batch_size)]

            if retriever == 'dpr':
                enc_inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = outputs.pooler_output.detach()
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings)
            elif retriever == 'contriever':
                enc_inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt'
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = mean_pooling(outputs[0], enc_inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings)
            elif retriever == 'dragon':
                enc_inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors='pt'
                ).to(device)
                outputs = encoder(**enc_inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings)
            else:
                raise ValueError(f"Unknown retriever type: {retriever}")

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


def build_faiss_index(
    embeddings: torch.Tensor,
    metric: str = 'ip'
):
    if faiss is None:
        raise ImportError("faiss is not installed, please `pip install faiss-gpu` or `faiss-cpu`.")

    if isinstance(embeddings, torch.Tensor):
        xb = embeddings.detach().cpu().numpy()
    else:
        xb = embeddings

    xb = xb.astype('float32')
    dim = xb.shape[1]

    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(xb)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    index.add(xb)
    return index


def faiss_knn_search(
    index,
    query_embeddings: torch.Tensor,
    top_k: int = 8,
    metric: str = 'ip'
):
    if isinstance(query_embeddings, torch.Tensor):
        xq = query_embeddings.detach().cpu().numpy()
    else:
        xq = query_embeddings

    xq = xq.astype('float32')
    if metric == 'ip':
        faiss.normalize_L2(xq)

    D, I = index.search(xq, top_k)
    return D, I


def get_sparse_retriever(
    text_chunks: List[str],
    retriever: str = 'bm25',
    num: int = 8,
):
    if Document is None or BM25Retriever is None or TFIDFRetriever is None:
        raise ImportError(
            "langchain_community or langchain_core is not installed. "
            "Install via `pip install langchain-community langchain-core`."
        )

    documents = [Document(page_content=text) for text in text_chunks]
    if retriever == 'bm25':
        retr = BM25Retriever.from_documents(documents, k=num)
    elif retriever == 'tf-idf':
        retr = TFIDFRetriever.from_documents(documents, k=num)
    else:
        raise ValueError(f"Unknown sparse retriever type: {retriever}")

    return retr


def sparse_neighborhood_search(
    retriever,
    query: str,
    text_chunks: List[str],
) -> List[int]:
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_indices = [text_chunks.index(doc.page_content) for doc in retrieved_docs]
    return retrieved_indices


if __name__ == '__main__':
    pass
