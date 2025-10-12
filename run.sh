### Parse dataset
python parse_trivia_qa.py --model='gemma-2b'
python parse_datasets.py --model='gemma-2b' --dataset='sciq'
python parse_datasets.py --model='gemma-2b' --dataset='nq'

### Generation
python generate_triviaqa.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=0.1 --num_beams=10 --top_p=1.0 --dataset='trivia_qa'
python generate_all_datasets.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=1.0 --num_beams=10 --top_p=1.0 --dataset='sciq'
python generate_all_datasets.py --num_generations_per_prompt=10 --model='gemma-2b' --fraction_of_data_to_use=0.5 --num_beams=10 --top_p=1.0 --dataset='nq'

### Likelihood and Semantic similarity
python get_semantic_similarities.py --generation_model='gemma-2b' --dataset={dataset_name}
python get_likelihoods.py --evaluation_model='gemma-2b' --generation_model='gemma-2b' --dataset={dataset_name}
python calculate_rouge.py --model='gemma-2b' --dataset={dataset_name}

# -- Our score: PRO
python pro.py --model='gemma-2b' --dataset={dataset_name}

# -- Other baselines
python get_semantic_density.py --generation_model='gemma-2b' --dataset={dataset_name} 
python compute_confidence.py --generation_model='gemma-2b' --evaluation_model='gemma-2b' --dataset={dataset_name} 
python analyze_results.py --dataset={dataset_name} --model='gemma-2b' 
python results_table_auroc.py --dataset={dataset_name}