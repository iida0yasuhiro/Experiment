#★★ 本実験 ラベル伝播でHerlevはクリークもランダムも一気に出すように修正！（9月6日作成）
import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.preprocessing import normalize
import networkx as nx
import math
from scipy.stats import mode
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

SEED = None
np.random.seed(SEED)
random.seed(SEED)

def run_experiment(noise_percentage):
    try:
        with open('./H_full_merged_data.json', 'r') as f:
            data = json.load(f)
        nodes = list(data.keys())
        feature_vectors = np.array(list(data.values()))
        print(f"データファイル './H_full_merged_data.json' を読み込みました。ノード数: {len(nodes)}")

    except FileNotFoundError:
        print("エラー: データファイル './H_full_merged_data.json' が見つからない。ファイルパスを要確認。")
        return None
    except Exception as e:
        print(f"エラー: データファイル読み込み中にエラーが発生しました - {e}")
        return None

    similarity_matrix = cosine_similarity(feature_vectors)
    print("コサイン類似度行列を計算しました。")

    G = nx.Graph()
    G.add_nodes_from(nodes)

    labels_list = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70
    original_labels_dict = {nodes[i]: labels_list[i] for i in range(len(nodes))}

    class_counts = Counter(original_labels_dict.values())
    print("Node counts per class in the original data:", dict(class_counts))

    changed_nodes = []

    print(f"\nInjecting approximately {noise_percentage*100:.2f}% noise into each category.")

    for class_label, count in class_counts.items():
        nodes_in_class = [node for node, label in original_labels_dict.items() if label == class_label]
        num_noisy_in_class = int(count * noise_percentage)

        if num_noisy_in_class > 0 and num_noisy_in_class <= len(nodes_in_class):
            noisy_nodes_in_class = random.sample(nodes_in_class, num_noisy_in_class)
            changed_nodes.extend(noisy_nodes_in_class)
        elif num_noisy_in_class > len(nodes_in_class):
            print(f"Warning: Calculated number of noisy nodes ({num_noisy_in_class}) for class '{class_label}' exceeds the number of nodes in the class ({len(nodes_in_class)}). All nodes in this class will be set as noisy.")
            changed_nodes.extend(nodes_in_class)
        else:
            print(f"Info: No nodes to inject noise into for class '{class_label}' ({num_noisy_in_class} calculated).")

    for node in G.nodes():
        if node in changed_nodes:
            original_label = original_labels_dict[node]
            all_possible_labels = list(class_counts.keys())
            possible_new_labels = [label for label in all_possible_labels if label != original_label]

            if possible_new_labels:
                new_label = random.choice(possible_new_labels)
                G.nodes[node]['label'] = new_label
            else:
                G.nodes[node]['label'] = original_label
        else:
            G.nodes[node]['label'] = original_labels_dict[node]

    print(f"Total of {len(changed_nodes)} nodes selected for noise injection.")
    actual_changed_count = sum(G.nodes[node]['label'] != original_labels_dict[node] for node in G.nodes())
    print(f"Number of nodes whose labels were actually changed: {actual_changed_count}")

    threshold = 0.74
    edges_added = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
                edges_added += 1
    print(f"類似度 {threshold} 以上のエッジを {edges_added} 個追加しました。")

    def label_spreading(graph, initial_labels, alpha=0.1, max_iter=100, tol=1e-5):
        unique_labels = sorted(list(set(initial_labels.values())))
        if not unique_labels:
            print(" initial_labelsが空です。ラベル伝播は実行されていない。")
            return {node: None for node in graph.nodes()}

        label_indices = {label: i for i, label in enumerate(unique_labels)}
        num_labels = len(unique_labels)
        num_nodes = len(graph)
        nodes_list = list(graph.nodes())

        F = np.zeros((num_nodes, num_labels))
        for node, label in initial_labels.items():
            if node in nodes_list:
                node_idx = nodes_list.index(node)
                if label in label_indices:
                    label_idx = label_indices[label]
                    F[node_idx, label_idx] = 1

        W = nx.adjacency_matrix(graph, nodelist=nodes_list).toarray()

        row_sum = W.sum(axis=1)
        D_inv_sqrt = np.diag(np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0))
        S = np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)

        Y = np.copy(F)

        for iter_count in range(max_iter):
            F_new = alpha * np.dot(S, F) + (1 - alpha) * Y
            diff = np.linalg.norm(F_new - F)
            F = F_new
            if diff < tol:
                print(f"ラベル伝播が収束しました (iteration: {iter_count+1})")
                break
        else:
            print(f"ラベル伝播が最大反復回数 ({max_iter}) に達したが収束せず。")

        new_labels_list = []
        for i in range(num_nodes):
            node = nodes_list[i]
            if node in initial_labels:
                new_labels_list.append(initial_labels[node])
            else:
                if np.sum(F[i, :]) > 0:
                    new_labels_list.append(unique_labels[np.argmax(F[i, :])])
                else:
                    new_labels_list.append(None)

        new_labels_dict = {nodes_list[i]: new_labels_list[i] for i in range(num_nodes)}
        return new_labels_dict

    def evaluate_results(propagated_labels, original_labels, changed_nodes, nodes):
        TP = sum(1 for node in changed_nodes if propagated_labels.get(node) == original_labels[node])
        FN = sum(1 for node in changed_nodes if propagated_labels.get(node) != original_labels[node] and propagated_labels.get(node) is not None)
        unchanged_nodes = [node for node in nodes if node not in changed_nodes]
        FP = sum(1 for node in unchanged_nodes if propagated_labels.get(node) != original_labels[node] and propagated_labels.get(node) is not None)
        TN = sum(1 for node in unchanged_nodes if propagated_labels.get(node) == original_labels[node])

        total_evaluated_nodes = TP + FN + FP + TN
        accuracy = (TP + TN) / total_evaluated_nodes if total_evaluated_nodes > 0 else 0

        return {'accuracy': accuracy}


    # --- クリークに基づく初期ラベル付け (新手法) ---
    def find_maximal_cliques(graph):
        maximal_cliques = []
        for clique in nx.find_cliques(graph):
            if len(clique) >= 3:
                maximal_cliques.append(set(clique))
        filtered_cliques = []
        for c1 in maximal_cliques:
            is_subset = False
            for c2 in maximal_cliques:
                if c1 != c2 and c1.issubset(c2):
                    is_subset = True
                    break
            if not is_subset:
                filtered_cliques.append(list(c1))
        return filtered_cliques

    maximal_cliques = find_maximal_cliques(G)
    sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

    initial_labels_clique = {}
    for clique in sorted_cliques:
        current_clique_labels = {node: G.nodes[node]['label'] for node in clique}
        label_weights = {}
        for node in clique:
            label = current_clique_labels[node]
            if label not in label_weights:
                label_weights[label] = 0
            for neighbor in G.neighbors(node):
                if neighbor in clique:
                    label_weights[label] += G[node][neighbor]['weight']

        dominant_label = None
        if label_weights:
            dominant_label = max(label_weights, key=label_weights.get)
        else:
            clique_labels_list = list(current_clique_labels.values())
            if clique_labels_list:
                dominant_label_mode, _ = mode(clique_labels_list, keepdims=True)
                dominant_label = dominant_label_mode[0]

        dominant_nodes_in_clique = [node for node in clique if current_clique_labels.get(node) == dominant_label]
        initial_node = None
        for node in dominant_nodes_in_clique:
            if node not in initial_labels_clique:
                initial_node = node
                break
        if initial_node is None and dominant_nodes_in_clique:
            initial_node = dominant_nodes_in_clique[0]

        if dominant_label is not None and initial_node is not None:
            initial_labels_clique[initial_node] = dominant_label

    clique_accuracy = None
    if initial_labels_clique:
        print(f"\nクリークの初期ラベルとして {len(initial_labels_clique)} 個のノードを選択。")
        propagated_labels_clique = label_spreading(G, initial_labels_clique)

        # 評価 (クリーク)
        y_true = []
        y_pred_clique = []
        for node in nodes:
            if propagated_labels_clique.get(node) is not None:
                y_true.append(original_labels_dict[node])
                y_pred_clique.append(propagated_labels_clique[node])

        clique_accuracy = accuracy_score(y_true, y_pred_clique)
        print("\n--- クリークに基づく初期ラベルを使用した場合の評価 ---")
        print(f"精度 (Accuracy): {clique_accuracy:.4f}")
    else:
        print("\nクリークに基づく初期ラベルが選択されなかったため、評価はスキップされます。")

    # --- ランダムな初期ラベル付け（97%） ---
    initial_labels_percentage = 0.97
    num_initial_nodes = int(len(nodes) * initial_labels_percentage)

    random_initial_labels = {}
    if len(nodes) >= num_initial_nodes:
        random_initial_nodes = random.sample(nodes, num_initial_nodes)
        for node in random_initial_nodes:
            if node in G.nodes():
                random_initial_labels[node] = G.nodes[node]['label']
    else:
        print("警告: ノード総数より初期ラベル数の方が多いため、ランダムな初期ラベルの選択をスキップします。")

    random_accuracy = None
    if random_initial_labels:
        print(f"\nランダムな初期ラベルとして {len(random_initial_labels)} 個のノードを選択しました。")
        random_propagated_labels = label_spreading(G, random_initial_labels)
        random_results = evaluate_results(random_propagated_labels, original_labels_dict, changed_nodes, nodes)
        random_accuracy = random_results['accuracy']
        print("\n--- ランダムな初期ラベルを使用した場合の評価 ---")
        print(f"精度 (Accuracy): {random_accuracy:.4f}")
    else:
        print("\nランダムな初期ラベルが選択されなかったため、評価はスキップ。")

    return {
        'noise_level': noise_percentage,
        'clique_accuracy': clique_accuracy,
        'random_accuracy': random_accuracy
    }

if __name__ == '__main__':
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    results = []

    for level in noise_levels:
        result = run_experiment(level)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        print("\n--- Summary of Labeling Results (Herlev) ---")
        print(df)