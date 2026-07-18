# 提案手法　アノテータバイアスモデル　7月5日作成、7月18日更新
import os
import json
import ast
import numpy as np
import pandas as pd
import pymc as pm
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data(feat_json, raw_noise_json, class_to_idx):
    with open(feat_json, 'r') as f:
        features = json.load(f)
    with open(raw_noise_json, 'r') as f:
        raw_data = ast.literal_eval(f.read())

    X = StandardScaler().fit_transform(np.array(list(features.values()))).astype(np.float32)
    name_to_idx = {os.path.splitext(os.path.basename(n))[0]: i for i, n in enumerate(features.keys())}

    num_images = X.shape[0]
    num_annotators = len(raw_data)
    num_classes = len(class_to_idx)

    ground_truth_labels = np.zeros(num_images, dtype=int)
    ann_ids, decisions, image_ids = [], [], []
    annotator_matrices = [np.zeros((num_classes, num_classes)) for _ in range(num_annotators)]

    for k, p_dict in enumerate(raw_data):
        for c_name, sub_dict in p_dict.items():
            for i_name, label in sub_dict.items():
                c_name_clean = os.path.splitext(os.path.basename(i_name))[0]
                if c_name_clean in name_to_idx:
                    img_idx = name_to_idx[c_name_clean]
                    ann_ids.append(k)
                    image_ids.append(img_idx)
                    decisions.append(int(label))

                    true_cls = class_to_idx[c_name]
                    ground_truth_labels[img_idx] = true_cls
                    annotator_matrices[k][true_cls, int(label)] += 1

    return X, np.array(ann_ids), np.array(image_ids), np.array(decisions), ground_truth_labels, annotator_matrices

def compute_graph_laplacian(X, knn=5):
    sim_matrix = cosine_similarity(X)
    W = np.zeros_like(sim_matrix)
    for i in range(len(X)):
        idx = np.argsort(sim_matrix[i])[-knn-1:-1]
        W[i, idx] = sim_matrix[i, idx]
        W[idx, i] = sim_matrix[i, idx]

    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, W

def estimate_biases_graph(X, ann_ids, image_ids, decisions, L, num_classes=6):
    print("\n[PyMC] グラフ正則化あり の変分推論 (30,000 steps) ---")
    num_annotators = len(np.unique(ann_ids))
    num_features = X.shape[1]
    num_images = X.shape[0]

    with pm.Model() as model:
        annotator_bias = pm.Normal('annotator_bias', mu=0, sigma=0.1, shape=(num_annotators, num_features))
        weights = pm.Normal('weights', mu=0, sigma=0.5, shape=(num_features, num_classes))
        graph_smoothness = pm.Normal('graph_smoothness', mu=0, sigma=0.2, shape=(num_images, num_classes))
        graph_penalty = pm.math.dot(pm.math.dot(graph_smoothness.T, L), graph_smoothness)
        pm.Potential('graph_regularizer', -0.5 * pm.math.sum(graph_penalty))

        deformed_features = X[image_ids] * annotator_bias[ann_ids]
        logits = pm.math.dot(deformed_features, weights) + graph_smoothness[image_ids]
        pm.Categorical('obs', p=pm.math.softmax(logits, axis=1), observed=decisions)

        approx = pm.fit(n=30000, method='advi', progressbar=True)
        trace = approx.sample(20)

    return (trace.posterior['annotator_bias'].mean(dim=['chain', 'draw']).values,
            trace.posterior['weights'].mean(dim=['chain', 'draw']).values,
            trace.posterior['graph_smoothness'].mean(dim=['chain', 'draw']).values)

def estimate_biases_no_graph(X, ann_ids, image_ids, decisions, num_classes=6):
    print("\n[PyMC] グラフ正則化なしの変分推論 (30,000 steps) ")
    num_annotators = len(np.unique(ann_ids))
    num_features = X.shape[1]

    with pm.Model() as model:
        annotator_bias = pm.Normal('annotator_bias', mu=0, sigma=0.1, shape=(num_annotators, num_features))
        weights = pm.Normal('weights', mu=0, sigma=0.5, shape=(num_features, num_classes))
        deformed_features = X[image_ids] * annotator_bias[ann_ids]
        logits = pm.math.dot(deformed_features, weights)
        pm.Categorical('obs', p=pm.math.softmax(logits, axis=1), observed=decisions)

        approx = pm.fit(n=30000, method='advi', progressbar=True)
        trace = approx.sample(20)

    return (trace.posterior['annotator_bias'].mean(dim=['chain', 'draw']).values,
            trace.posterior['weights'].mean(dim=['chain', 'draw']).values)

def compute_normalized_cm(gt_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes))
    for g, p in zip(gt_labels, pred_labels):
        cm[g, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums + 1e-9, out=np.zeros_like(cm), where=row_sums != 0)

def compute_cm_metrics(cm_true, cm_pred, num_classes):
    row_emds = []
    row_jsds = []
    class_values = np.arange(num_classes)

    for r in range(num_classes):
        p_true = cm_true[r]
        p_pred = cm_pred[r]

        if p_true.sum() == 0 or p_pred.sum() == 0:
            continue

        emd_val = wasserstein_distance(class_values, class_values, p_true, p_pred)
        row_emds.append(emd_val)

        m = 0.5 * (p_true + p_pred)
        jsd_val = 0.5 * entropy(p_true, m, base=2) + 0.5 * entropy(p_pred, m, base=2)
        row_jsds.append(jsd_val)

    mean_emd = np.mean(row_emds) if len(row_emds) > 0 else np.nan
    mean_jsd = np.mean(row_jsds) if len(row_jsds) > 0 else np.nan
    return mean_emd, mean_jsd

if __name__ == "__main__":
    class_to_idx = {'NILM': 0, 'ASC-US': 1, 'ASC-H': 2, 'LSIL': 3, 'HSIL': 4, 'SCC': 5}
    num_classes = len(class_to_idx)

    X, ann_ids, image_ids, decisions, ground_truth_labels, matrices = prepare_data(
        'SM-official-updated.json', 'raw_noise_patterns-all-1.json', class_to_idx
    )
    num_images, num_annotators = X.shape[0], len(matrices)

    L_laplacian, W_similarity = compute_graph_laplacian(X, knn=5)
    biases_g, W_g, G_g = estimate_biases_graph(X, ann_ids, image_ids, decisions, L_laplacian, num_classes)
    biases_ng, W_ng = estimate_biases_no_graph(X, ann_ids, image_ids, decisions, num_classes)

    obs_response_matrix = np.full((num_annotators, num_images), -1)
    for a, img, d in zip(ann_ids, image_ids, decisions):
        obs_response_matrix[a, img] = d

    comparison_results = []

    print("\n評価：行ごと混同行列のEMD & JSDを算出")
    for i in range(num_annotators):
        g_preds = np.zeros(num_images, dtype=int)
        ng_preds = np.zeros(num_images, dtype=int)

        for m in range(num_images):
            g_preds[m] = np.argmax(np.dot(X[m] * biases_g[i], W_g) + G_g[m])
            ng_preds[m] = np.argmax(np.dot(X[m] * biases_ng[i], W_ng))

        valid_idx = np.where(obs_response_matrix[i] != -1)[0]
        if len(valid_idx) == 0: continue

        y_true = obs_response_matrix[i, valid_idx]
        y_gt = ground_truth_labels[valid_idx]

        sub_g  = g_preds[valid_idx]
        sub_ng = ng_preds[valid_idx]

        cm_real = compute_normalized_cm(y_gt, y_true, num_classes)
        cm_g    = compute_normalized_cm(y_gt, sub_g, num_classes)
        cm_ng   = compute_normalized_cm(y_gt, sub_ng, num_classes)

        g_emd,  g_jsd  = compute_cm_metrics(cm_real, cm_g, num_classes)
        ng_emd, ng_jsd = compute_cm_metrics(cm_real, cm_ng, num_classes)

        comparison_results.append({
            'Annotator': f'User_{i}',
            'GG_EMD': g_emd,   'GG_JSD': g_jsd,
            'GNG_EMD': ng_emd, 'GNG_JSD': ng_jsd
        })

    df_comp = pd.DataFrame(comparison_results)

    cols = ['Annotator', 'GG_EMD', 'GG_JSD', 'GNG_EMD', 'GNG_JSD']

    print("\n" + "="*125)
    print("  【論文実験用：2手法分布近似精度評価結果】")
    print("   GG: GEMINI(Graph) | GNG: GEMINI(No-Graph)")
    print("   EMD、JSD ともに、値が低いほどよい")
    print("="*125)
    print(df_comp[cols].to_string(index=False, formatters={c: lambda x: f"{x:.3f}" for c in cols if c != 'Annotator'}))
    print("-"*125)

    print(f"【全体平均】 GEMINI_G  (グラフあり) -> CM_EMD: {df_comp['GG_EMD'].mean():.3f} | CM_JSD: {df_comp['GG_JSD'].mean():.3f}")
    print(f"【全体平均】 GEMINI_NG (グラフなし) -> CM_EMD: {df_comp['GNG_EMD'].mean():.3f} | CM_JSD: {df_comp['GNG_JSD'].mean():.3f}")
    print("=============================================================================================================\n")

    np.save('estimated_biases.npy', biases_g)
    np.save('annotator_matrices.npy', np.array(matrices))
    print("[完了（EMD / JSD）を出力済。")