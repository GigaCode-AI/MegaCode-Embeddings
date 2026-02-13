# Setup

```commandline
uv sync
```

1. Unpack models; change `model.model_dir` in `inference/config/experiment/retriever/megacode-emb*.yaml`
2. Unpack `benchmarks.zip`

# Training

- [Pretraining script](`train/train_retriever_wo_hard_neg.py`)
- [Contrastive finetuning script](train/train_retriever_with_hard_neg.py)
- [Contrastive + KL finetuning script](train/train_retriever_hybrid.py)
- [KL finetuning script](train/train_retriever_kl.py)

# Inference

- [Predict script](infrence/jobs/predict.sh)
- [Annotate with qwen3-32b in reasoning mode](infrence/jobs/annotate.sh)  
- [Compute metrics and compare models notebook](infrence/evaluate.ipynb)
