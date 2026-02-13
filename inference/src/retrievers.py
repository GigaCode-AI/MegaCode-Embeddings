import re
from typing import List

import tqdm
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM
from vllm.inputs.data import TokensPrompt
from vllm.config import PoolerConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer


class BaseRetriever:
    def vectorize(self, texts: List[str], max_tokens: int, use_tqdm: bool):
        """
        Returns torch.tensor(..., dtype=torch.float32, device="cuda:0")
        """
        raise NotImplementedError


class RetrieverVLLM(BaseRetriever):
    def __init__(self, model_dir: str, eos_token_id: int, m: int = None):
        """
        m - how many leading dimensions to take. Can be non-None only when training in matryoshka mode
        """
        pooler_config = PoolerConfig(pooling_type="LAST", normalize=False)
        self.model = LLM(model_dir, task="embed", override_pooler_config=pooler_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = eos_token_id
        self.m = m

    def vectorize(self, texts, max_tokens=512, use_tqdm=True) -> torch.Tensor:
        input_ids = self.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_tokens - 1)["input_ids"]
        inputs = [TokensPrompt(prompt_token_ids=x + [self.eos_token_id]) for x in input_ids]
        outputs = self.model.embed(inputs, use_tqdm=use_tqdm)
        # x.outputs.embedding is List[float]; slice [:None] is equivalent to taking the full list
        emb = torch.tensor([x.outputs.embedding[:self.m] for x in outputs], dtype=torch.float32, device="cuda:0")
        F.normalize(emb, out=emb)
        return emb


class RetrieverCodeSage(BaseRetriever):
    def __init__(self, pretrained_dir):
        self.model = AutoModel.from_pretrained(pretrained_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, trust_remote_code=True, add_eos_token=True)
        self.device = "cuda:0"
        self.model.to(self.device).eval()

    def vectorize(self, texts, max_tokens=512, use_tqdm=True) -> torch.Tensor:
        # * max_tokens=2048 from model config
        # * Embedding should be taken this way; the HF example only shows how to get hidden states:
        # https://github.com/amazon-science/CodeSage/blob/main/evaluation/nl2code_search.py#L83
        bs = 4
        n = len(texts)
        X = torch.zeros((n, self.model.config.hidden_size), dtype=torch.float32, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16), tqdm.tqdm(total=n, disable=not use_tqdm) as bar:
            for i in range(0, n, bs):
                j = min(i + bs, n)
                inputs = self.tokenizer(
                    texts[i:j],
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                X[i:j] = F.normalize(outputs.pooler_output)
                bar.update(j - i)
        return X


class RetrieverJinaV2(BaseRetriever):
    def __init__(self, pretrained_dir: str):
        self.model = AutoModel.from_pretrained(pretrained_dir, trust_remote_code=True)
        self.model.to("cuda:0").eval()

    def vectorize(self, texts, max_tokens=512, use_tqdm=True) -> torch.Tensor:
        # Doing exactly as in the HF example:
        # https://huggingface.co/jinaai/jina-embeddings-v2-base-code#usage
        with torch.autocast("cuda", dtype=torch.float16):
            X = self.model.encode(
                texts,
                batch_size=8,
                max_length=max_tokens,
                show_progress_bar=use_tqdm,
                normalize_embeddings=True,
                convert_to_tensor=True,
                convert_to_numpy=False
            )
        return X


class RetrieverJinaV4(BaseRetriever):
    """
    https://huggingface.co/jinaai/jina-embeddings-v4-vllm-code
    mean pooling + norm
    """
    def __init__(self, pretrained_dir: str):
        self.model = LLM(
            model=pretrained_dir,
            task="embed",
            override_pooler_config=PoolerConfig(pooling_type="MEAN", normalize=True),
            dtype="float16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

    def vectorize(self, texts, max_tokens=512, use_tqdm=True) -> torch.Tensor:
        # Verified: special tokens are not added when doing this
        input_ids = self.tokenizer(texts, truncation=True, max_length=max_tokens - 1)["input_ids"]
        inputs = [TokensPrompt(prompt_token_ids=x) for x in input_ids]
        outputs = self.model.embed(inputs, use_tqdm=use_tqdm)
        res = torch.tensor([x.outputs.embedding for x in outputs], dtype=torch.float32, device="cuda:0")
        return res


class RetrieverCornStack(BaseRetriever):
    def __init__(self, pretrained_dir: str):
        # bfloat16:
        # https://github.com/gangiswag/cornstack/blob/main/src/evaluations/eval_csn.py#L28
        self.model = SentenceTransformer(pretrained_dir, trust_remote_code= True).to("cuda:0").to(torch.bfloat16)

    def vectorize(self, texts: List[str], max_tokens: int = 512, use_tqdm: bool = True):
        self.model.max_seq_length = max_tokens
        # For some reason the HF example doesn't use cosine:
        # https://huggingface.co/nomic-ai/CodeRankEmbed
        # Paper says they use cosine:
        # https://arxiv.org/pdf/2412.01007, section 3.1
        res = self.model.encode(
            texts,
            show_progress_bar=use_tqdm,
            normalize_embeddings=True,
            batch_size=8,
            convert_to_tensor=True,
            device="cuda:0"
        )
        return res.to(torch.float32)


def split_camel_case(s: str) -> List[str]:
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+', s)
    if len(parts) > 0:
        return parts
    return [s]


def analyzer_v1(text: str):
    """
    1-3 grams over words, lowercase
    """
    tokens = []
    for w in re.findall("\w+", text.lower()):
        tokens.append(w)
    res = []
    for k in [1, 2, 3]:
        for i in range(len(tokens) - k + 1):
            res.append(" ".join(tokens[i:i + k]))
    return res


def analyzer_v2(text: str) -> List[str]:
    """
    Like v1, but "weird" words are split into tokens (e.g. FooBar -> [Foo, Bar], read_data -> [read, data])
    """
    tokens = []
    for w in re.findall("\w+", text):
        for t in split_camel_case(w):
            tokens.append(t.lower())  # important to lowercase here!
    res = []
    for k in [1, 2, 3]:
        for i in range(len(tokens) - k + 1):
            res.append(" ".join(tokens[i:i + k]))
    return res


class RetrieverTfidf:
    def __init__(self, analyzer):
        self.vec = TfidfVectorizer(analyzer=analyzer)

    def fit(self, texts: List[str]):
        self.vec.fit(texts)

    def transform(self, texts: List[str]) -> csr_matrix:
        return self.vec.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        self.fit(texts)
        return self.transform(texts)


if __name__ == "__main__":
    for case in ["fooBar", "FooBar", "foo_bar", "foo_Bar", "1", "foo", "Bar", "foo-1", "йцукен", "Йцукен"]:
        print(case, split_camel_case(case))
