# Reproducibility Guidelines

## 🛠️ Environment Setup

Create a conda environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate pro-env
```
Ensure that the appropriate model weights (e.g., `gemma-2b`) are available locally or accessible via Hugging Face.

## 🗂️ 1. Parse Datasets
Preprocess and cache input prompts for each dataset:
```bash
# TriviaQA
python parse_trivia_qa.py --model='gemma-2b'

# SciQ and Natural Questions
python parse_datasets.py --model='gemma-2b' --dataset='sciq'
python parse_datasets.py --model='gemma-2b' --dataset='nq'
```

## 🧠 2. Beam Search Generation
Generate multiple completions per prompt using beam search.

```bash
# TriviaQA (10% of data)
python generate_triviaqa.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=0.1 \
  --num_beams=10 --top_p=1.0 \
  --dataset='trivia_qa'

# SciQ (full dataset)
python generate_all_datasets.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=1.0 \
  --num_beams=10 --top_p=1.0 \
  --dataset='sciq'

# NQ (50% of data)
python generate_all_datasets.py \
  --num_generations_per_prompt=10 \
  --model='gemma-2b' \
  --fraction_of_data_to_use=0.5 \
  --num_beams=10 --top_p=1.0 \
  --dataset='nq'
```

## 📊 3. Evaluation and Analysis
🔍 Semantic Similarity and Likelihood
```bash
# Semantic similarities
python get_semantic_similarities.py --generation_model='gemma-2b' --dataset={dataset_name}

# Likelihood
python get_likelihoods.py --evaluation_model='gemma-2b' --generation_model='gemma-2b' --dataset={dataset_name}

# RougeL for correctness computation
python calculate_rouge.py --model='gemma-2b' --dataset={dataset_name}
```

## ✅ PRO Score (Ours)
```bash
python pro.py --model='gemma-2b' --dataset={dataset_name}
```

## 📉 Baselines for Comparison
```bash
# Semantic Density
python get_semantic_density.py --generation_model='gemma-2b' --dataset={dataset_name}

# Other baselines
python compute_confidence.py --generation_model='gemma-2b' --evaluation_model='gemma-2b' --dataset={dataset_name}

# Final results
python analyze_results.py --dataset={dataset_name} --model='gemma-2b'

python results_table_auroc.py --dataset={dataset_name}
```

## 📓 Example Workflow
```bash
bash run.sh
```
