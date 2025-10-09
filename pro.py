import argparse
import os
import pickle
import tqdm
import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import config
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Mistral-7B-v0.1')
parser.add_argument('--dataset', type=str, default='coqa')
args = parser.parse_args()

# Set a seed value
seed_value = 3
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def approx(probs):
    pk = probs[-1]
    score = -np.log(pk) - np.sum([pi * np.log(pi / pk) for pi in probs[:-1]])
    
    return score

num_beams = 10
model = args.model
dataset = args.dataset

with open(f'{config.output_dir}/{model}_generations_all_{dataset}.pkl', 'rb') as infile:
    generations = pickle.load(infile)

with open(f'{config.output_dir}/{model}_generations_{model}_likelihoods_all_{dataset}_temperature1.0.pkl', 'rb') as infile:
    likelihoods = pickle.load(infile)

df = pd.DataFrame() # to store output scores
for i in tqdm.tqdm(range(len(generations))):
    
    # Load saved generations
    generation = generations[i]
    likelihood = likelihoods[i]
            
    df.loc[i, 'label'] = 1 - (generation[f'rougeL_beam_search_to_target_0'] > 0.3).astype('int')     
    df.loc[i, 'avg_nll'] = likelihood['average_neg_log_likelihood_of_most_likely_gen'].item() # for verification
    df.loc[i, 'nll'] = likelihood['neg_log_likelihood_of_most_likely_gen'].item() # for verification

    for beam_id in range(num_beams):
        beam_generation_ids = generation[f'cleaned_beam_search_generation_{beam_id}_ids']
        beam_generation_text = generation[f'cleaned_beam_search_generation_{beam_id}']

        # Extract nll of generation
        df.loc[i, f'avg_nll_{beam_id}'] = likelihood[f'average_neg_log_likelihood_of_beam_search_gen_{beam_id}'].item()
        df.loc[i, f'nll_{beam_id}'] = likelihood[f'neg_log_likelihood_of_beam_search_gen_{beam_id}'].item()

df.to_csv(f'{config.result_dir}/pro_raw_{model}_{dataset}.csv', index=False)
df = df.dropna()   
     
# ---Fine-tuning
alpha_list = np.arange(0, 1.01, 0.05)
        
for i in range(num_beams):
    df[f'p{i}'] = df[f'nll_{i}'].apply(lambda x: np.exp(-x))

tmp_df = pd.DataFrame()
p_cols = [f'p{i}' for i in range(10)]            

for i, alpha in enumerate(alpha_list):
    tmp_df.loc[i, 'dataset'] = dataset
    tmp_df.loc[i, 'model'] = model
    tmp_df.loc[i, 'baseline'] = roc_auc_score(df['label'], df['nll'])
    tmp_df.loc[i, 'alpha'] = alpha
    
    # ---Adaptive select top K
    cnt = 0
    list_s1, list_s2 = [], []
    list_k = []
    for idx in df.index:
        sample = df.loc[idx, p_cols]
        
        # ---Sort & filter top probs: p0 > p1 > p2 <=> nll_0 < nll_1 < nll_2
        top_probs = np.array(sorted(sample, reverse=True))
        probs = top_probs[top_probs > alpha]

        # ---Replace with p0 in case no prob
        if len(probs) < 1:
            probs = [df.loc[idx, 'p0']]
            cnt += 1
        
        s1 = approx(probs=probs)

        list_s1.append(s1)
        list_k.append(len(probs))
    
    tmp_df.loc[i, 'num_of_no_prob'] = cnt
    tmp_df.loc[i, 'num_of_records'] = df.shape[0]
    tmp_df.loc[i, '%_of_no_prob'] = round(cnt / df.shape[0], 2)
    tmp_df.loc[i, 'nll_approx'] = roc_auc_score(df['label'], list_s1)                
    tmp_df.loc[i, 'k'] = np.mean(list_k)

tmp_df.to_csv(f'{config.result_dir}/tmp_pro_{model}_{dataset}.csv', index=False)

# ---Select the max score
final_df = tmp_df[['model', 'dataset', 'baseline', 'nll_approx']].groupby(['model', 'dataset']).max().reset_index()
final_df.to_csv(f'{config.result_dir}/pro_{model}_{dataset}.csv', index=False)
