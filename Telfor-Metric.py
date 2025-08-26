#ViT Herlev Discussion向けエッジ閾値依存性　ConvNextでも同じ結果になりそう
#Netwrokxのクリークカウント関数はバグがありそうだから、使わない。
import json
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy.stats import entropy
import gc
import random

def calculate_simple_entropy(clique_nodes, graph_nodes):
    """
    クリーク内のラベル分布から単純なエントロピーを計算する関数。
    """
    label_counts = Counter(graph_nodes[node]['label'] for node in clique_nodes)
    entropy_val = 0
    num_nodes = len(clique_nodes)

    if num_nodes == 0:
        return 0

    for count in label_counts.values():
        probability = count / num_nodes
        if probability > 0:
            entropy_val -= probability * math.log2(probability)

    return entropy_val

def calculate_jsd(p, q):
    """
    2つの確率分布pとqのJSダイバージェンスを計算する。
    """
    p = np.array(p) + 1e-9
    q = np.array(q) + 1e-9

    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def modify_calculate_nkd_entropy(clique_nodes, graph_nodes, total_label_probs):
    """
    与えられた数式 H_NKD(C) に基づいてエントロピーを計算する関数
    (JSダイバージェンスを乗算するように修正)
    """
    num_nodes_in_clique = len(clique_nodes)

    if num_nodes_in_clique <= 1:
        return 0

    clique_label_counts = Counter(graph_nodes[node]['label'] for node in clique_nodes)

    all_unique_labels = sorted(list(total_label_probs.keys()))

    p_dist_list = [total_label_probs.get(label, 0) for label in all_unique_labels]
    p_dist = np.array(p_dist_list)

    q_dist_list = [clique_label_counts.get(label, 0) / num_nodes_in_clique for label in all_unique_labels]
    q_dist = np.array(q_dist_list)

    jsd_value = calculate_jsd(p_dist, q_dist)

    sum_term = 0

    for label, count in clique_label_counts.items():
        p_l_given_c = count / num_nodes_in_clique

        if p_l_given_c > 0:
            term = jsd_value * p_l_given_c * math.log2(p_l_given_c)
            sum_term += term

    log_denominator = math.log2(num_nodes_in_clique)

    if log_denominator > 0:
        nkd_entropy = - (1 / log_denominator) * sum_term
        return nkd_entropy
    else:
        return 0


def find_maximal_cliques(graph):
    """
    グラフ内の極大クリークをすべて見つけるため自作した関数。Networkxの関数はバグがあるので決して使わないこと
    """
    maximal_cliques = []
    node_list = list(graph.nodes())
    for node in node_list:
        clique = {node}
        neighbors = set(graph.neighbors(node))
        candidates = neighbors.copy()
        for other_node in clique:
            candidates &= set(graph.neighbors(other_node))
        while candidates:
            new_node = candidates.pop()
            clique.add(new_node)
            is_clique = True
            for existing_node in clique:
                if new_node != existing_node and not graph.has_edge(new_node, existing_node):
                    is_clique = False
                    break
            if is_clique:
                neighbors = set(graph.neighbors(new_node))
                new_candidates = neighbors.copy()
                for other_node in clique:
                    new_candidates &= set(graph.neighbors(other_node))
                candidates &= new_candidates
            else:
                clique.remove(new_node)
        is_maximal = True
        for existing_clique in maximal_cliques:
            if clique.issubset(existing_clique):
                is_maximal = False
                break
        if is_maximal:
            is_subset = False
            for other_clique in maximal_cliques:
                if clique != other_clique and clique.issubset(other_clique):
                    is_subset = True
                    break
            if not is_subset:
                maximal_cliques.append(clique)
    unique_cliques = [set(t) for t in set(tuple(sorted(list(s))) for s in maximal_cliques)]
    return unique_cliques


def run_analysis(data, labels_list, threshold, noise_percentage=0):
    """
    指定されたデータ、ラベル、ノイズ率でグラフ分析を実行する関数
    """
    print(f"\n--- ノイズ率 {noise_percentage*100:.0f}% での分析実行 ---")
    nodes = list(data.keys())
    current_labels_list = labels_list.copy()

    if noise_percentage > 0:
        num_to_noise = int(len(current_labels_list) * noise_percentage)
        indices_to_noise = random.sample(range(len(current_labels_list)), num_to_noise)

        all_labels = list(set(current_labels_list))
        for i in indices_to_noise:
            original_label = current_labels_list[i]
            possible_labels = [l for l in all_labels if l != original_label]
            if possible_labels:
                current_labels_list[i] = random.choice(possible_labels)

    feature_vectors = np.array(list(data.values()))
    similarity_matrix = cosine_similarity(feature_vectors)
    del feature_vectors
    gc.collect()

    G = nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(node, label=current_labels_list[i])

    total_label_counts = Counter(current_labels_list)
    total_nodes = len(current_labels_list)
    total_label_probs = {label: count / total_nodes for label, count in total_label_counts.items()}

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(nodes[i], nodes[j])

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"エッジ数: {G.number_of_edges()}")
    print(f"孤立ノード削除後のノード数: {G.number_of_nodes()}")
    del similarity_matrix
    gc.collect()

    print("極大クリークを計算...")
    all_cliques = find_maximal_cliques(G)
    print(f"見つかった極大クリークの総数: {len(all_cliques)}個")

    min_sizes = range(3, 15)
    simple_entropies = []
    modified_nkd_entropies = []

    for min_clique_size in min_sizes:
        simple_clique_entropies = []
        modified_nkd_clique_entropies = []

        for clique in all_cliques:
            if len(clique) >= min_clique_size:
                simple_entropy = calculate_simple_entropy(clique, G.nodes)
                modified_nkd_entropy = modify_calculate_nkd_entropy(clique, G.nodes, total_label_probs)

                simple_clique_entropies.append(simple_entropy)
                modified_nkd_clique_entropies.append(modified_nkd_entropy)

        if simple_clique_entropies:
            simple_entropies.append(np.mean(simple_clique_entropies))
            modified_nkd_entropies.append(np.mean(modified_nkd_clique_entropies))
        else:
            simple_entropies.append(0)
            modified_nkd_entropies.append(0)

    return min_sizes, simple_entropies, modified_nkd_entropies

# --- メイン処理 ---
try:
    with open('./H_full_merged_data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Herlevのデータファイルがない場合のエラー回避のみ。")
    num_nodes = 917
    nodes = [f'node_{i}' for i in range(num_nodes)]
    feature_vectors_dummy = np.random.rand(num_nodes, 10)
    data = {nodes[i]: feature_vectors_dummy[i].tolist() for i in range(num_nodes)}

labels_list_original = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70
num_runs = 10 #試行回数（信頼性区間）

# --- 最初の設定：ノイズ率を変化させる場合 ---
results_by_noise = {}
noise_definitions = {
    '0%_noise': 0.0,
    '10%_noise': 0.1,
    '20%_noise': 0.2,
    '30%_noise': 0.3,
    '40%_noise': 0.4,
    '50%_noise': 0.5
}
threshold_fixed = 0.74

print("--- ノイズ率変化の計算開始 ---")
for name, noise_percentage in noise_definitions.items():
    run_simple_entropies = []
    run_modified_nkd_entropies = []

    print(f"--- 分析: {name} ---")
    for i in range(num_runs):
        print(f"--- 試行 {i+1}/{num_runs} ---")
        min_sizes, simple_e, modified_nkd_e = run_analysis(data, labels_list_original, threshold_fixed, noise_percentage=noise_percentage)
        run_simple_entropies.append(simple_e)
        run_modified_nkd_entropies.append(modified_nkd_e)

    results_by_noise[name] = {
        'simple_mean': np.mean(run_simple_entropies, axis=0),
        'simple_std': np.std(run_simple_entropies, axis=0),
        'modified_nkd_mean': np.mean(run_modified_nkd_entropies, axis=0),
        'modified_nkd_std': np.std(run_modified_nkd_entropies, axis=0),
    }
print("--- ノイズ率変化の終了 ---")

# --- 最初のプロット：min_sizes vs. Average Entropy ---
plt.figure(figsize=(14, 10))
min_sizes = range(3, 15)

# 既存のラベルエントロピーのプロット
plt.plot(min_sizes, results_by_noise['0%_noise']['simple_mean'], marker='o', linestyle='-', color='b', label='Simple Entropy (Original)')
plt.errorbar(min_sizes, results_by_noise['10%_noise']['simple_mean'], yerr=results_by_noise['10%_noise']['simple_std'],
             marker='s', linestyle='--', color='g', label='Simple Entropy (10% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['20%_noise']['simple_mean'], yerr=results_by_noise['20%_noise']['simple_std'],
             marker='^', linestyle=':', color='r', label='Simple Entropy (20% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['30%_noise']['simple_mean'], yerr=results_by_noise['30%_noise']['simple_std'],
             marker='x', linestyle='-', color='olive', label='Simple Entropy (30% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['40%_noise']['simple_mean'], yerr=results_by_noise['40%_noise']['simple_std'],
             marker='*', linestyle='--', color='teal', label='Simple Entropy (40% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['50%_noise']['simple_mean'], yerr=results_by_noise['50%_noise']['simple_std'],
             marker='D', linestyle=':', color='lime', label='Simple Entropy (50% Noise)', capsize=5)

# 提案方式のプロット
plt.plot(min_sizes, results_by_noise['0%_noise']['modified_nkd_mean'], marker='o', linestyle='-', color='purple', label='Modified NKD Entropy (Original)')
plt.errorbar(min_sizes, results_by_noise['10%_noise']['modified_nkd_mean'], yerr=results_by_noise['10%_noise']['modified_nkd_std'],
             marker='s', linestyle='--', color='orange', label='Modified NKD Entropy (10% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['20%_noise']['modified_nkd_mean'], yerr=results_by_noise['20%_noise']['modified_nkd_std'],
             marker='^', linestyle=':', color='brown', label='Modified NKD Entropy (20% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['30%_noise']['modified_nkd_mean'], yerr=results_by_noise['30%_noise']['modified_nkd_std'],
             marker='x', linestyle='-', color='cyan', label='Modified NKD Entropy (30% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['40%_noise']['modified_nkd_mean'], yerr=results_by_noise['40%_noise']['modified_nkd_std'],
             marker='*', linestyle='--', color='magenta', label='Modified NKD Entropy (40% Noise)', capsize=5)
plt.errorbar(min_sizes, results_by_noise['50%_noise']['modified_nkd_mean'], yerr=results_by_noise['50%_noise']['modified_nkd_std'],
             marker='D', linestyle=':', color='gray', label='Modified NKD Entropy (50% Noise)', capsize=5)

plt.xlabel('Minimum Clique Size (N)', fontsize=12)
plt.ylabel('Average Label Entropy', fontsize=12)
plt.title('Comparison of Average Entropy Metrics with Label Noise', fontsize=14)
plt.xticks(min_sizes)
plt.grid(True)
plt.legend(loc='best', fontsize=8)
plt.show()

# --- 新しい分析：thresholdを変化させる場合 (min_size=3に固定) ---
results_by_threshold = {}
threshold_definitions = [0.72, 0.73, 0.74, 0.75, 0.76]
min_size_fixed = 3
noise_fixed = 0.0

print("\n--- Threshold変化の分析開始 (min_size=3に固定) ---")
for threshold in threshold_definitions:
    run_simple_entropies = []
    run_modified_nkd_entropies = []

    print(f"--- 分析: Threshold={threshold} ---")
    for i in range(num_runs):
        print(f"--- 試行 {i+1}/{num_runs} ---")
        min_sizes_result, simple_e, modified_nkd_e = run_analysis(data, labels_list_original, threshold, noise_percentage=noise_fixed)

        # min_size=3の結果のみを抽出
        min_size_index = list(min_sizes_result).index(min_size_fixed)
        run_simple_entropies.append(simple_e[min_size_index])
        run_modified_nkd_entropies.append(modified_nkd_e[min_size_index])

    results_by_threshold[threshold] = {
        'simple_mean': np.mean(run_simple_entropies),
        'simple_std': np.std(run_simple_entropies),
        'modified_nkd_mean': np.mean(run_modified_nkd_entropies),
        'modified_nkd_std': np.std(run_modified_nkd_entropies),
    }
print("--- Threshold変化の分析終了 ---")


# --- 新しいプロット：Threshold vs. Average Entropy (for min_size=3) ---
plt.figure(figsize=(10, 8))

simple_means_at_min3 = [results_by_threshold[t]['simple_mean'] for t in threshold_definitions]
simple_stds_at_min3 = [results_by_threshold[t]['simple_std'] for t in threshold_definitions]
modified_nkd_means_at_min3 = [results_by_threshold[t]['modified_nkd_mean'] for t in threshold_definitions]
modified_nkd_stds_at_min3 = [results_by_threshold[t]['modified_nkd_std'] for t in threshold_definitions]

plt.errorbar(threshold_definitions, simple_means_at_min3, yerr=simple_stds_at_min3,
             marker='o', linestyle='-', color='b', label='Simple Entropy', capsize=5)

plt.errorbar(threshold_definitions, modified_nkd_means_at_min3, yerr=modified_nkd_stds_at_min3,
             marker='s', linestyle='--', color='purple', label='Modified NKD Entropy', capsize=5)

plt.xlabel('Similarity Threshold', fontsize=12)
plt.ylabel('Metric Value (for min clique size=3)', fontsize=12)
plt.title('Impact of Similarity Threshold on Entropy Metrics (min clique size=3)', fontsize=14)
plt.xticks(threshold_definitions)
plt.grid(True)
plt.legend()
plt.show()