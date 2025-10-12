import argparse
import pickle
import torch
from sklearn.metrics import roc_auc_score
import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
args = parser.parse_args()

model_name_list = ['Mixtral-8x7B-v0.1', 'Meta-Llama-3-8B', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B', 'Mistral-7B-v0.1', 'Mixtral-8x7B-v0.1', 'Mixtral-8x22B-v0.1']
with open(f"{config.paper_result_dir}/Table_auroc_{args.model}_{args.dataset}_temperature{args.temperature}.txt", "a+") as text_file:
    print(args.dataset, file=text_file)

with open(f'{config.result_dir}/{args.model}_p_true_{args.dataset}.pkl', 'rb') as outfile:
    p_trues_across_beams, labels_across_beams = pickle.load(outfile)

    p_trues_beam0 = p_trues_across_beams[0]
    corrects_beam0 = labels_across_beams[0]

    p_true_auroc_all = roc_auc_score(1 - torch.tensor(corrects_beam0), torch.tensor(p_trues_beam0))

with open(f'{config.result_dir}/overall_results_beam_search_{args.model}_{args.dataset}_temperature1.0.pkl', 'rb') as f:
    overall_result_dict = pickle.load(f)
result_dict = overall_result_dict['beam_0']


table_row = '{}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(
                                                    args.model, result_dict['semantic_density_auroc'], result_dict['entropy_over_concepts_auroc'],
                                                    p_true_auroc_all, result_dict['average_contradict_prob_auroc'], result_dict['average_neg_log_likelihood_auroc'],
                                                    result_dict['ln_predictive_entropy_auroc'], result_dict['predictive_entropy_auroc'], result_dict['average_rougeL_auroc'])
with open(f"{config.paper_result_dir}/Table_auroc_{args.model}_{args.dataset}_temperature{args.temperature}.txt", "a+") as text_file:
    print(table_row, file=text_file)

