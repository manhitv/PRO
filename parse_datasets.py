import argparse
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from transformers import AutoTokenizer
import config
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--few_shot_num', type=int, default=10)
args = parser.parse_args()
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

question_file = f'{config.data_dir}/{args.dataset}/test.txt'

mistralai_models = ['Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
llama_models = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Llama-3.1-8B', 'Llama-3.2-1B', 'Llama-3.2-3B']
opt_models = ['opt-6.7b', 'opt-13b', 'opt-30b', 'opt-2.7b', 'opt-1.3b']
gemma_models = ['gemma-2-9b', 'gemma-2b', 'gemma-7b', 'gemma-2-2b', 'gemma-2-27b']
falcon_models = ['falcon-7b-instruct', 'Falcon3-7B-Instruct', 'Falcon3-10B-Instruct', 'falcon-7b', 'falcon-40b', 'falcon-11b']
    
if f"{args.model}" in mistralai_models:
    hf_model_dir = 'mistralai/' + f"{args.model}"

if f"{args.model}" in llama_models:
    hf_model_dir = 'meta-llama/' + f"{args.model}"

if f"{args.model}" in opt_models:
    hf_model_dir = 'facebook/' + f"{args.model}"

if f"{args.model}" in gemma_models:
    hf_model_dir = 'google/' + f"{args.model}"

if f"{args.model}" in falcon_models:
    hf_model_dir = 'tiiuae/' + f"{args.model}"


tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=False, cache_dir=config.hf_cache_dir)

seed_value = 10

print('Preprocessing dataset')
if question_file.endswith('.txt'):
    questions = []
    answers = []
    for item in open(question_file).read().split('\n\n'):
        if item:
            questions.append(item.split('\n')[0])
            answers.append(item.split('\n')[1].split(';'))
        answers[-1] = [answer.strip() for answer in answers[-1]]

elif question_file.endswith('.csv'):
    data_df = pd.read_csv(question_file)
    questions = data_df['question'].tolist()
    answers = data_df['answer'].tolist()
else:
    print('question file format must be txt or csv')
    exit()

few_shot_prompt = 'This is a bot that correctly answers questions. \n'
for i in range(args.few_shot_num):
    few_shot_prompt += 'Question: ' + questions[i] + ' Answer: ' + answers[i][0] + ' '

processed_dataset = []
for i in range(args.few_shot_num, len(questions)):
    processed_data = {}
    processed_data['question'] = questions[i]
    processed_data['answer'] = answers[i]
    prompt = few_shot_prompt + "Question: " + questions[i] + " Answer:"
    processed_data['prompt'] = prompt
    inputs = tokenizer(prompt, padding=False, truncation=False)
    outputs = tokenizer(answers[i], padding=False, truncation=False)
    processed_data["input_ids"] = inputs.input_ids
    processed_data["attention_mask"] = inputs.attention_mask
    processed_data["answer_input_ids"] = outputs.input_ids
    processed_data["answer_attention_mask"] = outputs.attention_mask
    processed_data["labels"] = outputs.input_ids.copy()
    processed_data["labels"] = [-100 if token == tokenizer.pad_token_id else token for token in processed_data["labels"]]

    processed_dataset.append(processed_data)

with open(f'{config.data_dir}/{args.dataset}_{args.model}.pkl', 'wb') as dataset_file:
    pickle.dump(processed_dataset, dataset_file)
