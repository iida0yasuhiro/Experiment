# ★★★5月26日　論文最終掲載　（アノテーションセッション１）
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# --- 定数・ハイパーパラメータ ---
ALPHA = 0.2
SIMILARITY_THRESHOLD = 0.86
INITIAL_LABEL_RATIO = 0.1
TOLERANCE = 1e-6
MAX_ITER = 100
NUM_TRIALS = 20  # 各アノテータ数条件でのアンサンブル回数

# ★ 3回のシード変更計測用
SEED_LIST = [42, 100, 2026]
NUM_TOTAL_ANNOTATORS = 11
IMAGES_PER_ANNOTATOR = 105
NUM_CLASSES = 6
FEATURE_DIM = 768
EPSILON = 1e-12

LABEL_MAPPING = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
REV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# SLF用ハイパーパラメータ
SLF_HYPERPARAMS = {
    'lr': 0.0001,
    'epochs': 20,
    'batch_size': 64,
    'lambda': 1.0,
    'M': 30
}

def build_graph(features, similarity_threshold):
    similarity_matrix = cosine_similarity(features)
    W = similarity_matrix * (similarity_matrix >= similarity_threshold)
    return W

def get_true_labels(file_names):
    true_label_map = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
    label_mapping = {
        'NILM': 'NILM', 'ASCUS': 'ASC-US', 'LSIL': 'LSIL',
        'ASCH': 'ASC-H', 'HSIL': 'HSIL', 'SCC': 'SCC'
    }
    true_labels = []
    for fname in file_names:
        name_without_ext = fname.split('.')[0]
        prefix = ''.join(filter(str.isalpha, name_without_ext))
        mapped_prefix = label_mapping.get(prefix, prefix)
        true_labels.append(true_label_map[mapped_prefix])
    return np.array(true_labels)

def label_propagation(Y0, W, max_iter, tolerance):
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    Y = Y0.copy()
    for i in range(max_iter):
        Y_prev = Y.copy()
        Y = ALPHA * S @ Y + (1 - ALPHA) * Y0
        if np.linalg.norm(Y - Y_prev) < tolerance:
            break
    sum_Y = Y.sum(axis=1, keepdims=True)
    return np.divide(Y, sum_Y, out=np.zeros_like(Y), where=sum_Y != 0)

def apply_confusion_matrix_correction(Y_final, confusion_matrix):
    conf_matrix_prob = confusion_matrix / 100
    Y_updated = np.dot(Y_final, conf_matrix_prob)
    return Y_updated / (Y_updated.sum(axis=1, keepdims=True) + 1e-12)


# --- 1. Dawid-Skeneアルゴリズム モジュール ---
def initialize_ds_parameters(data_df, all_items):
    filtered_df = data_df[data_df['item_id'].isin(all_items)].copy()
    annotators = filtered_df['annotator_id'].unique()
    num_items = len(all_items)

    item_labels_mv = filtered_df.groupby('item_id')['label'].agg(lambda x: x.value_counts().index[0])
    class_counts = item_labels_mv.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    class_priors = (class_counts.values + 1) / (num_items + NUM_CLASSES)

    annotator_errors = {k: np.zeros((NUM_CLASSES, NUM_CLASSES)) for k in annotators}
    item_true_labels_map = item_labels_mv.to_dict()

    for row in filtered_df.itertuples():
        true_label = item_true_labels_map.get(row.item_id)
        if true_label is not None:
            annotator_errors[row.annotator_id][int(true_label), int(row.label)] += 1

    for k in annotators:
        annotator_errors[k] += 1
        row_sums = annotator_errors[k].sum(axis=1, keepdims=True)
        annotator_errors[k] = np.divide(annotator_errors[k], row_sums,
                                        out=np.full_like(annotator_errors[k], 1/NUM_CLASSES),
                                        where=row_sums!=0)
    return class_priors, annotator_errors

def run_dawid_skene(data_df, all_items, max_iter=20, tol=1e-4):
    annotators = data_df['annotator_id'].unique()
    num_items = len(all_items)
    class_priors, annotator_errors = initialize_ds_parameters(data_df, all_items)

    data_map = {item: {} for item in all_items}
    for row in data_df.itertuples():
        if row.item_id in data_map:
            data_map[row.item_id][row.annotator_id] = row.label

    for _ in range(max_iter):
        item_probas = np.zeros((num_items, NUM_CLASSES))
        for idx, item_id in enumerate(all_items):
            log_probas = np.log(class_priors.copy() + 1e-12)
            if item_id in data_map and data_map[item_id]:
                for j in range(NUM_CLASSES):
                    for annotator_id, label in data_map[item_id].items():
                        error_matrix = annotator_errors[annotator_id]
                        if error_matrix[j, label] > 1e-9:
                            log_probas[j] += np.log(error_matrix[j, label])
            max_log = np.max(log_probas)
            probas_exp = np.exp(log_probas - max_log)
            item_probas[idx] = probas_exp / np.sum(probas_exp)

        class_priors = item_probas.sum(axis=0) / num_items

        for annotator_id in annotators:
            error_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
            annotator_data = data_df[data_df['annotator_id'] == annotator_id]
            for row in annotator_data.itertuples():
                if row.item_id in all_items:
                    item_idx = all_items.index(row.item_id)
                    error_matrix[:, row.label] += item_probas[item_idx]

            annotated_items = [itm for itm in annotator_data['item_id'].unique() if itm in all_items]
            if len(annotated_items) > 0:
                annotated_indices = [all_items.index(itm) for itm in annotated_items]
                sum_T_ij = item_probas[annotated_indices].sum(axis=0)
                annotator_errors[annotator_id] = np.divide(error_matrix, sum_T_ij[:, np.newaxis],
                                                           out=np.full_like(error_matrix, 1/NUM_CLASSES),
                                                           where=sum_T_ij[:, np.newaxis] > 1e-9)
    return item_probas


# --- 2. Sample-wise Label Fusion (SLF) モジュール ---
class NoisyLabelsDataset(Dataset):
    def __init__(self, features, noisy_labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(noisy_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LabelFusionNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_annotators, num_basis_matrices, hidden_dim=128):
        super(LabelFusionNet, self).__init__()
        self.num_annotators = num_annotators
        self.num_basis_matrices = num_basis_matrices

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.prediction_head = nn.Linear(hidden_dim, num_classes)
        self.weights_head = nn.Linear(hidden_dim, num_annotators)
        self.coeffs_head = nn.Linear(hidden_dim, num_annotators * num_basis_matrices)

    def forward(self, x):
        shared_features = self.backbone(x)
        class_logits = self.prediction_head(shared_features)
        annotator_weights = torch.softmax(self.weights_head(shared_features), dim=1)
        confusion_coeffs = self.coeffs_head(shared_features).view(-1, self.num_annotators, self.num_basis_matrices)
        confusion_coeffs = torch.softmax(confusion_coeffs, dim=2)
        return class_logits, annotator_weights, confusion_coeffs

class SLFTrainer:
    def __init__(self, all_items, df_annotations, features_matrix, valid_fnames_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_fnames = valid_fnames_list

        annotator_list = sorted(df_annotations['annotator_id'].unique())
        self.annotator_map = {name: i for i, name in enumerate(annotator_list)}
        num_annotators = len(self.annotator_map)

        df_annotations = df_annotations.copy()
        df_annotations['annotator_idx'] = df_annotations['annotator_id'].map(self.annotator_map)

        pivoted = df_annotations.pivot(index='item_id', columns='annotator_idx', values='label')
        noisy_labels_df = pivoted.reindex(self.valid_fnames).fillna(-1)

        valid_indices = [all_items.index(f) for f in self.valid_fnames]
        self.sub_features = features_matrix[valid_indices]
        self.sub_noisy_labels = noisy_labels_df.values

        self.model = LabelFusionNet(
            input_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
            num_annotators=num_annotators, num_basis_matrices=SLF_HYPERPARAMS['M']
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=SLF_HYPERPARAMS['lr'])
        self.basis_matrices = self._generate_permutation_matrices(SLF_HYPERPARAMS['M'], NUM_CLASSES).to(self.device)

    def _generate_permutation_matrices(self, M, K):
        matrices = torch.zeros(M, K, K)
        identity = torch.eye(K)
        for i in range(M):
            matrices[i] = identity[torch.randperm(K)]
        return matrices

    def _compute_loss(self, logits, weights, coeffs, noisy_labels):
        P = torch.einsum('brm,mkl->brkl', coeffs, self.basis_matrices)
        y_noisy_onehot = F.one_hot(noisy_labels.clamp(min=0), num_classes=NUM_CLASSES).float()
        y_clean = torch.matmul(P, y_noisy_onehot.unsqueeze(-1)).squeeze(-1)
        mask = (noisy_labels != -1).float().unsqueeze(-1)

        y_targ = torch.sum(weights.unsqueeze(-1) * y_clean * mask, dim=1) + EPSILON
        y_targ = y_targ / y_targ.sum(dim=1, keepdim=True)

        log_preds = F.log_softmax(logits, dim=1)
        loss_kl = F.kl_div(log_preds, y_targ, reduction='batchmean')

        diag_P = torch.diagonal(P, dim1=-2, dim2=-1)
        reg_term = torch.sum((1.0 - diag_P)**2, dim=2)
        loss_reg = torch.mean(torch.sum(reg_term * mask.squeeze(-1), dim=1) / (mask.sum(dim=1).clamp(min=EPSILON)))
        return loss_kl + SLF_HYPERPARAMS['lambda'] * loss_reg

    def fit_and_predict(self):
        dataset = NoisyLabelsDataset(self.sub_features, self.sub_noisy_labels)
        dataloader = DataLoader(dataset, batch_size=SLF_HYPERPARAMS['batch_size'], shuffle=True)

        # モデルの訓練
        self.model.train()
        for epoch in range(SLF_HYPERPARAMS['epochs']):
            for feats, labels in dataloader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits, weights, coeffs = self.model(feats)
                loss = self._compute_loss(logits, weights, coeffs, labels)
                if torch.isnan(loss):
                    return np.zeros((len(self.valid_fnames), NUM_CLASSES))
                loss.backward()
                self.optimizer.step()

        # 全サンプルの推論(Softmax確率を出力)
        self.model.eval()
        eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_probas = []
        with torch.no_grad():
            for feats, _ in eval_loader:
                feats = feats.to(self.device)
                logits, _, _ = self.model(feats)
                probas = F.softmax(logits, dim=1)
                all_probas.extend(probas.cpu().numpy())
        return np.array(all_probas)


# --- 3. 実験ランナーループ ---
def run_experiment(num_active_annotators, annotator_workloads, all_initial_labels, features, true_labels, file_names_list, confusion_matrix):
    num_nodes = len(file_names_list)
    active_indices = random.sample(range(NUM_TOTAL_ANNOTATORS), num_active_annotators)

    active_fnames = set()
    for idx in active_indices:
        active_fnames.update(annotator_workloads[idx])
    valid_fnames_this_run = list(active_fnames)
    total_eval = len(valid_fnames_this_run)

    if total_eval == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0

    ensemble_Y_corrected = np.zeros((num_nodes, 6))
    W = build_graph(features, SIMILARITY_THRESHOLD)

    # 共通アノテーションプール（DSおよびSLF用インプット）
    annotations_pool = []

    for _ in range(NUM_TRIALS):
        random.shuffle(valid_fnames_this_run)
        num_initial = int(total_eval * INITIAL_LABEL_RATIO)
        selected_fnames = set(valid_fnames_this_run[:num_initial])

        Y0 = np.zeros((num_nodes, 6))
        for i, fname in enumerate(file_names_list):
            if fname in selected_fnames:
                Y0[i] = all_initial_labels[fname]
                simulated_label = np.argmax(all_initial_labels[fname])
                for a_idx in active_indices:
                    if fname in annotator_workloads[a_idx]:
                        annotations_pool.append({
                            'item_id': fname, 'annotator_id': f"User_{a_idx}", 'label': simulated_label
                        })

        sum_Y0 = Y0.sum(axis=1, keepdims=True)
        Y0 = np.divide(Y0, sum_Y0, out=np.zeros_like(Y0), where=sum_Y0 != 0)

        Y_res = label_propagation(Y0, W, MAX_ITER, TOLERANCE)
        ensemble_Y_corrected += apply_confusion_matrix_correction(Y_res, confusion_matrix)

    # A. 提案手法（グラフ+補正）の精度計算
    prop_counts = [0, 0, 0]
    for fname in valid_fnames_this_run:
        idx = file_names_list.index(fname)
        true_class = true_labels[idx]
        pred_rank = np.argsort(ensemble_Y_corrected[idx])[::-1]
        if true_class == pred_rank[0]: prop_counts[0] += 1
        if true_class in pred_rank[:2]: prop_counts[1] += 1
        if true_class in pred_rank[:3]: prop_counts[2] += 1
    prop_accuracies = [(c / total_eval * 100) for c in prop_counts]

    # データフレーム化（重複削除）
    df_annotations = pd.DataFrame(annotations_pool).drop_duplicates(subset=['item_id', 'annotator_id'])

    # B. Dawid-Skene の精度計算
    ds_accuracies = [0.0, 0.0, 0.0]
    if len(df_annotations) > 0:
        ds_probas = run_dawid_skene(df_annotations, valid_fnames_this_run)
        ds_counts = [0, 0, 0]
        for f_idx, fname in enumerate(valid_fnames_this_run):
            orig_idx = file_names_list.index(fname)
            true_class = true_labels[orig_idx]
            ds_rank = np.argsort(ds_probas[f_idx])[::-1]
            if true_class == ds_rank[0]: ds_counts[0] += 1
            if true_class in ds_rank[:2]: ds_counts[1] += 1
            if true_class in ds_rank[:3]: ds_counts[2] += 1
        ds_accuracies = [(c / total_eval * 100) for c in ds_counts]

    # C. Sample-wise Label Fusion (SLF) の精度計算
    slf_accuracies = [0.0, 0.0, 0.0]
    if len(df_annotations) > 0:
        slf_engine = SLFTrainer(file_names_list, df_annotations, features, valid_fnames_this_run)
        slf_probas = slf_engine.fit_and_predict()
        slf_counts = [0, 0, 0]
        for f_idx, fname in enumerate(valid_fnames_this_run):
            orig_idx = file_names_list.index(fname)
            true_class = true_labels[orig_idx]
            slf_rank = np.argsort(slf_probas[f_idx])[::-1]
            if true_class == slf_rank[0]: slf_counts[0] += 1
            if true_class in slf_rank[:2]: slf_counts[1] += 1
            if true_class in slf_rank[:3]: slf_counts[2] += 1
        slf_accuracies = [(c / total_eval * 100) for c in slf_counts]

    return prop_accuracies, ds_accuracies, slf_accuracies, total_eval

def main():
    try:
        with open('SM-official-updated.json', 'r', encoding='utf-8') as f:
            sm_data = {k.replace('./', ''): v for k, v in json.load(f).items()}
        file_names_list = list(sm_data.keys())
        features = np.array(list(sm_data.values()))
        true_labels = get_true_labels(file_names_list)

        raw_initial_labels = {}
        with open('error_distribution_vectors.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if "ノード（ファイル）:" in lines[i]:
                    fname = lines[i].split(': ')[1].strip().replace('-', '') + '.jpg'
                    vector_str = lines[i+1].split('[')[1].split(']')[0]
                    raw_initial_labels[fname] = np.fromstring(vector_str, sep=' ')

        common_fnames = [f for f in file_names_list if f in raw_initial_labels]
    except Exception as e:
        print(f"エラー: {e}")
        return

    confusion_matrix = np.array([
        [67.5, 23.4, 8.4, 0.6, 0.0, 0.0],
        [1.4, 15.8, 19.2, 13.7, 37.7, 12.3],
        [0.7, 14.4, 41.8, 11.6, 16.4, 15.1],
        [0.0, 5.0, 6.3, 13.2, 49.1, 26.4],
        [2.0, 8.4, 10.1, 14.5, 48.8, 16.2],
        [0.8, 6.7, 5.1, 16.2, 39.9, 31.2]
    ])

    annotator_counts = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    results_summary = {
        count: {
            'prop_top1': [], 'prop_top2': [], 'prop_top3': [],
            'ds_top1': [], 'ds_top2': [], 'ds_top3': [],
            'slf_top1': [], 'slf_top2': [], 'slf_top3': [],
            'n_sizes': []
        } for count in annotator_counts
    }

    print(f"計 {len(SEED_LIST)} 回の独立試行を開始します...")
    for seed_idx, seed in enumerate(SEED_LIST):
        print(f"  [試行 {seed_idx+1}/{len(SEED_LIST)}] Seed: {seed} を実行中...")
        random.seed(seed)
        np.random.seed(seed)

        annotator_workloads = []
        for _ in range(NUM_TOTAL_ANNOTATORS):
            workload = random.sample(common_fnames, IMAGES_PER_ANNOTATOR)
            annotator_workloads.append(workload)

        for count in annotator_counts:
            prop_accs, ds_accs, slf_accs, n_size = run_experiment(
                count, annotator_workloads, raw_initial_labels,
                features, true_labels, file_names_list, confusion_matrix
            )
            results_summary[count]['prop_top1'].append(prop_accs[0])
            results_summary[count]['prop_top2'].append(prop_accs[1])
            results_summary[count]['prop_top3'].append(prop_accs[2])
            results_summary[count]['ds_top1'].append(ds_accs[0])
            results_summary[count]['ds_top2'].append(ds_accs[1])
            results_summary[count]['ds_top3'].append(ds_accs[2])
            results_summary[count]['slf_top1'].append(slf_accs[0])
            results_summary[count]['slf_top2'].append(slf_accs[1])
            results_summary[count]['slf_top3'].append(slf_accs[2])
            results_summary[count]['n_sizes'].append(n_size)

    # --- 最終集計結果テーブル出力 ---
    print("\n" + "="*125)
    print(f"3手法ベンチマーク比較一覧 (Mean [Max - Min 差分], 独立試行回数: {len(SEED_LIST)})")
    print("="*125)
    print(f"{'アノテータ':<5} | {'手法':<6} | {'平均母数':<6} | {'Top-1 精度 (%)':<18} | {'Top-2 精度 (%)':<18} | {'Top-3 精度 (%)':<18}")
    print("-" * 125)

    for count in annotator_counts:
        metrics = results_summary[count]
        mean_n = np.mean(metrics['n_sizes'])

        for mode in ['prop', 'ds', 'slf']:
            if mode == 'prop': name = "提案"
            elif mode == 'ds': name = "DS"
            else: name = "SLF"

            m_t1 = metrics[f'{mode}_top1']
            m_t2 = metrics[f'{mode}_top2']
            m_t3 = metrics[f'{mode}_top3']

            mean_t1, diff_t1 = np.mean(m_t1), np.max(m_t1) - np.min(m_t1)
            mean_t2, diff_t2 = np.mean(m_t2), np.max(m_t2) - np.min(m_t2)
            mean_t3, diff_t3 = np.mean(m_t3), np.max(m_t3) - np.min(m_t3)

            t1_str = f"{mean_t1:>6.2f} [{diff_t1:<5.2f}]"
            t2_str = f"{mean_t2:>6.2f} [{diff_t2:<5.2f}]"
            t3_str = f"{mean_t3:>6.2f} [{diff_t3:<5.2f}]"

            if mode == 'prop':
                print(f"{count:>4}名   | {name:<4} | {mean_n:>5.1f}枚 | {t1_str:<18} | {t2_str:<18} | {t3_str:<18}")
            else:
                print(f"      | {name:<4} |        | {t1_str:<18} | {t2_str:<18} | {t3_str:<18}")
        print("-" * 125)
    print("="*125)

if __name__ == "__main__":
    main()