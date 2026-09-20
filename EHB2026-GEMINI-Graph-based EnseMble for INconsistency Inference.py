#EHB2026投稿GEMINI 5月26日
import numpy as np
import os
import json
import random
import ast
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import pointbiserialr, entropy

# --- 1. 設定 & パラメータ ---
SEED = 0
NUM_TRIALS = 20
threshold = 0.8
alpha_val = 0.2
epsilon = 0.001
#W_ATTEN = 9.5  # GEMINI-b_weight 用に維持

label_map_6 = {0: 'NILM', 1: 'ASC-US', 2: 'LSIL', 3: 'ASC-H', 4: 'HSIL', 5: 'SCC'}
categories_6 = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
pattern_names = ["AK0326", "HA0523", "MI0722", "SI0417", "SN1017", "SS0122", "SS19750114", "TK0317", "TM0725", "TM0815", "YH0309"]

# --- 2. グラフ構築（連結性補完） ---
target_file = 'SM-official-updated.json'
with open(target_file, 'r') as f:
    data = json.load(f)

nodes = [os.path.splitext(os.path.basename(node))[0] for node in data.keys()]
feature_vectors = np.array(list(data.values()))
sim_matrix = cosine_similarity(feature_vectors)

W_bin = np.where(sim_matrix > threshold, 1, 0)
np.fill_diagonal(W_bin, 0)
G = nx.from_numpy_array(W_bin)
W = np.where(sim_matrix > threshold, sim_matrix, 0.0)

components = list(nx.connected_components(G))
if len(components) > 1:
    main_comp = list(components[0])
    for i in range(1, len(components)):
        target_comp = list(components[i])
        u, v = main_comp[0], target_comp[0]
        W[u, v] = epsilon
        W[v, u] = epsilon

np.fill_diagonal(W, 0)
row_sum = W.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
S = D_inv_sqrt @ W @ D_inv_sqrt

# --- 3. 分析・比較用関数 ---

def label_spreading_dist(initial_map, S_matrix, alpha):
    Y = np.zeros((len(nodes), len(categories_6)))
    for node, label in initial_map.items():
        if node in nodes: Y[nodes.index(node), categories_6.index(label)] = 1.0
    F = np.copy(Y)
    for _ in range(100):
        F_new = alpha * (S_matrix @ F) + (1 - alpha) * Y
        if np.linalg.norm(F_new - F) < 1e-5: break
        F = F_new
    row_sums = F.sum(axis=1, keepdims=True)
    return np.divide(F, row_sums, out=np.zeros_like(F), where=row_sums!=0)

def compute_wsw_scores(all_avg_Fs, nodes, categories_6):
    global_mean_F = np.mean(all_avg_Fs, axis=0)
    annotator_weights = []
    for f in all_avg_Fs:
        pred_global = np.argmax(global_mean_F, axis=1)
        pred_local  = np.argmax(f, axis=1)
        weight = np.mean(pred_global == pred_local)
        annotator_weights.append(weight)

    annotator_weights = np.array(annotator_weights)
    annotator_weights /= (annotator_weights.sum() + 1e-10)

    wsw_F = np.zeros((len(nodes), len(categories_6)))
    for w, f in zip(annotator_weights, all_avg_Fs):
        wsw_F += w * f
    return wsw_F

def estimate_ds_weights_em(all_avg_Fs, nodes, categories_6, n_iter=10):
    n_nodes, n_classes = len(nodes), len(categories_6)
    T = np.mean(all_avg_Fs, axis=0)
    for _ in range(n_iter):
        pi = []
        for f in all_avg_Fs:
            confusion = np.zeros((n_classes, n_classes))
            for c in range(n_classes):
                mask = T[:, c]
                confusion[c] += (mask[:, None] * f).sum(axis=0)
            row_sum = confusion.sum(axis=1, keepdims=True)
            confusion = np.divide(confusion, row_sum, out=np.zeros_like(confusion), where=row_sum!=0)
            pi.append(confusion)

        log_T = np.zeros((n_nodes, n_classes))
        for f, p in zip(all_avg_Fs, pi):
            for c in range(n_classes):
                log_T[:, c] += np.log(np.clip((f * p[c]).sum(axis=1), 1e-10, None))
        T = np.exp(log_T)
        T /= (T.sum(axis=1, keepdims=True) + 1e-10)
    return T

def get_indiv_results(dict_source, dict_target, avg_F, matl_F, slf_entropy, wsw_F, ds_T):
    dup_nodes = set(dict_source.keys()) & set(dict_target.keys())
    res_list = []
    for node in dup_nodes:
        if node not in nodes: continue
        idx = nodes.index(node)
        src_label = dict_source[node]
        label_idx = categories_6.index(src_label)

        # 基礎計算
        p_res = avg_F[idx][label_idx]
        attenuation = 1.0 - p_res
        is_conf = (dict_source[node] != dict_target[node])
        p_src = np.zeros(6); p_src[label_idx] = 1.0

        # --- Base JSD ---
        js = jensenshannon(p_src + 1e-10, avg_F[idx] + 1e-10)       # GEMINI-a
        m_js = jensenshannon(p_src + 1e-10, matl_F[idx] + 1e-10)   # GEMINI-c Base
        w_js = jensenshannon(p_src + 1e-10, wsw_F[idx] + 1e-10)    # GEMINI-d Base

        # --- GEMINI-b_weight (重み付き減衰) ---
        score_weighted = js * (1 + W_ATTEN * attenuation)

        # --- その他指標 ---
        s_ent = slf_entropy[idx]                                   # SLF
        d_js = jensenshannon(p_src + 1e-10, ds_T[idx] + 1e-10)     # DS-based

        res_list.append({
            'js': js,
            'score_weighted': score_weighted, # 復活
            'matl_js': m_js,
            'wsw_js': w_js,
            'slf_score': s_ent,
            'ds_js': d_js,
            'conf': is_conf
        })
    return res_list

# --- 4. 実行 & データ集計 ---
def load_noise(file):
    if not os.path.exists(file): return []
    with open(file, 'r', encoding='utf-8') as f:
        return ast.literal_eval(f.read())

p1_raw = load_noise('raw_noise_patterns-all-1.json')
p2_raw = load_noise('raw_noise_patterns-all-2.json')

all_avg_F1, all_avg_F2 = [], []
all_dicts_1, all_dicts_2 = [], []
raw_counts1, raw_counts2 = np.zeros((len(nodes), 6)), np.zeros((len(nodes), 6))

for i in range(len(pattern_names)):
    d1 = {n: label_map_6[int(v)] for c, m in p1_raw[i].items() for n, v in m.items()}
    d2 = {n: label_map_6[int(v)] for c, m in p2_raw[i].items() for n, v in m.items()}
    all_dicts_1.append(d1); all_dicts_2.append(d2)

    for node, label in d1.items():
        if node in nodes: raw_counts1[nodes.index(node), categories_6.index(label)] += 1
    for node, label in d2.items():
        if node in nodes: raw_counts2[nodes.index(node), categories_6.index(label)] += 1

    def run_ls(d):
        acc = np.zeros((len(nodes), 6))
        for t in range(NUM_TRIALS):
            random.seed(SEED + t)
            acc += label_spreading_dist({n: d[n] for n in random.sample(list(d.keys()), int(len(d)*0.8))}, S, alpha_val)
        return acc / NUM_TRIALS

    all_avg_F1.append(run_ls(d1))
    all_avg_F2.append(run_ls(d2))

matl_F1, matl_F2 = np.mean(all_avg_F1, axis=0), np.mean(all_avg_F2, axis=0)
wsw_F1 = compute_wsw_scores(all_avg_F1, nodes, categories_6)
wsw_F2 = compute_wsw_scores(all_avg_F2, nodes, categories_6)
ds_T1 = estimate_ds_weights_em(all_avg_F1, nodes, categories_6)
ds_T2 = estimate_ds_weights_em(all_avg_F2, nodes, categories_6)

# Pure-DS の処理は無効化のまま

slf_e1 = np.array([entropy(r + 1e-10) for r in raw_counts1])
slf_e2 = np.array([entropy(r + 1e-10) for r in raw_counts2])

raw_results = {"1->2": [], "2->1": []}
for i in range(len(pattern_names)):
    raw_results["1->2"].append(get_indiv_results(all_dicts_1[i], all_dicts_2[i], all_avg_F1[i], matl_F1, slf_e1, wsw_F1, ds_T1))
    raw_results["2->1"].append(get_indiv_results(all_dicts_2[i], all_dicts_1[i], all_avg_F2[i], matl_F2, slf_e2, wsw_F2, ds_T2))

# --- 5. レポート出力 ---
# GEMINI-b_weight をリスト内に残して整理
metrics = [
    ("GEMINI-a", 'js'),
    ("GEMINI-b", 'score_weighted'), # 復活
    ("GEMINI-c", 'matl_js'),
    ("GEMINI-d", 'wsw_js'),
    ("Entropy-based", 'slf_score'),
    ("EM-based", 'ds_js')
]

print("\n" + "="*125)
print(f"STATE-OF-THE-ART ENSEMBLE COMPARISON: EXTENDED GEMINI VARIANTS")
print("="*125)

for key in ["1->2", "2->1"]:
    print(f"\n[Direction: {key}]")
    print(f"  {'Range':<15} | {'Metric':<15} | {'Corr (R)':<10} | {'p-value':<10} | {'Significant'} | {'Rank-1 Success'}")
    print(f"  {'-'*120}")
    # Top-1 (1) を除外した指定レンジ
    for k in [3, 5, 10, None]:
        first_row = True
        for m_name, m_key in metrics:
            all_s, all_l = [], []
            r1_sum, n_total = 0, 0
            for res in raw_results[key]:
                if not res: continue
                n_total += 1
                res.sort(key=lambda x: x[m_key], reverse=True)
                if res[0]['conf']: r1_sum += 1
                limit = k if k else len(res)
                for r in res[:limit]:
                    all_s.append(r[m_key])
                    all_l.append(1 if r['conf'] else 0)
            if all_l:
                if len(set(all_l)) < 2 or len(set(all_s)) < 2:
                    r_val, p_val, sig = 0.0, 1.0, "No"
                else:
                    r_val, p_val = pointbiserialr(all_l, all_s)
                    sig = "Yes" if p_val < 0.05 else "No"

                range_txt = (f"Top-{k}" if k else "All Samples") if first_row else ""
                print(f"  {range_txt:<15} | {m_name:<15} | {r_val:10.4f} | {p_val:.2e} | {sig:<11} | {r1_sum}/{n_total}")
                first_row = False
        print(f"  {'-'*120}")