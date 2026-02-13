export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="32"

versions=(
"qwen--qwen3-emb-06b"
"qwen--qwen3-emb-4b"
"qwen--qwen3-emb-8b"
"codesage--codesage-large-v2"
"jinaai--jina-embeddings-v2-base-code"
"infly--inf-retriever-v1-1.5b"
"infly--inf-retriever-v1"
"jinaai--jina-embeddings-v4-vllm-code"
"nomic-ai--CodeRankEmbed"
"tf-idf--default-analyzer"
"tf-idf--custom-analyzer"
"megaccode-emb-v1-0.5b-pt2"
"megaccode-emb-v1-1.5b-pt"
"megaccode-emb-v1-3b-pt"
"megaccode-emb-v1-7b-pt"
"megaccode-emb-v1-0.5b"
"megaccode-emb-v1-1.5b"
"megaccode-emb-v1-3b"
"megaccode-emb-v1-7b"
)

output_dir="retrieval_predictions"  # new folder for predictions
benches_dir="benchmarks"  # folder with benches

mkdir -p $output_dir

for v in "${versions[@]}"; do
  model_dir="${output_dir}/$v"
  mkdir -p "$model_dir"
  python predict_cosqa.py +experiment="retriever/${v}" data_path="${benches_dir}/cosqa" output_path="${model_dir}/cosqa.jsonl"
  python predict_cosqa.py +experiment="retriever/${v}" data_path="${benches_dir}/cosqa_plus" output_path="${model_dir}/cosqa-plus.jsonl"
  python predict_repoqa.py +experiment="retriever/${v}" data_path="${benches_dir}/repoqa-2024-06-23__with_parsed_query.json" output_path="${model_dir}/repoqa.jsonl"
  python predict_csn_advtest.py +experiment="retriever/${v}" data_path="${benches_dir}/codexglue_advtest/test.jsonl" output_path="${model_dir}/csn-adv.jsonl"
  python predict_csn_original.py +experiment="retriever/${v}" data_path="${benches_dir}/code_search_net/data" output_path="${model_dir}/csn-orig.jsonl"
  python predict_csn_query.py +experiment="retriever/${v}" data_path="${benches_dir}/csn_query_function" output_path="${model_dir}/csn-query.jsonl"
done
