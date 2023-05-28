import shap
import numpy as np
import lightgbm as lgb
from scipy.stats import kendalltau

import os
import argparse

from surrogate import get_ember_data, get_sorel_data


def find_num_common_elements(a, b):
    return len(np.intersect1d(a, b))

def get_shapley_indices(model_path, X_test, k):
    model = lgb.Booster(model_file=model_path)
    model.params["objective"] = 'binary'
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)[0]
    sort_inds = np.argsort(np.sum(np.abs(shap_values), axis=0))

    return sort_inds[::-1][:k]


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/mari/ember2018')
    parser.add_argument('--model_path', type=str, default='malware_rl/envs/utils')
    parser.add_argument('--num_shapley_indices', type=int, default=10)
    parser.add_argument('--num_test_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=26871)
    parser.add_argument('--target', choices=['ember', 'sorel'], default='ember', help='Target model')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    num_shapley_indices = args.num_shapley_indices
    num_test_samples = args.num_test_samples
    seed = args.seed
    target = args.target 

    np.random.seed(seed)

    if target == 'ember':
        _, X_test, _, _ = get_ember_data(data_dir)
    else:
        _, X_test, _, _ = get_sorel_data(data_dir)

    idx = np.random.choice(np.arange(X_test.shape[0]), num_test_samples)

    if target =='ember':
        # model_idx = get_shapley_indices(os.path.join(model_path, 'ember_model.txt'), X_test[idx], num_shapley_indices)
        ember_idx =  [691, 2359, 637, 655, 626, 930, 620, 2355, 683, 2364, 615, 2360, 2351, 2354, 692, 786, 2353, 511, 613, 2363]
        model_idx = ember_idx[:num_shapley_indices]
    else:
        model_idx = get_shapley_indices(os.path.join(model_path, 'sorel.model'), X_test[idx], num_shapley_indices)
    print(model_idx)

    surr_idx = get_shapley_indices(os.path.join(model_path, f'lgb_{target}_model_{seed}.txt'), X_test[idx], num_shapley_indices)
    print(surr_idx)

    print(find_num_common_elements(model_idx, surr_idx)/num_shapley_indices)
    print(kendalltau(model_idx, surr_idx))
    