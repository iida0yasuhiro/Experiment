# Sipakmed　を読み込んでSimRankを計算

import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform # Import pdist and squareform
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import directed_hausdorff
from collections import Counter

# JSONファイル(950個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./S_merged_data.json', 'r') as f:
    data = json.load(f)

# 特徴ベクトルをNumPy配列に変換
feature_vectors = np.array(list(data.values()))


# Hausdorff距離行列を計算
num_images = feature_vectors.shape[0]
hausdorff_matrix = np.zeros((num_images, num_images))
for i in range(num_images):
    for j in range(i + 1):
        # 対称行列なので、上三角行列のみ計算
        # Reshape feature vectors to 2D arrays for directed_hausdorff
        hausdorff_matrix[i, j] = directed_hausdorff(feature_vectors[i].reshape(1, -1),
                                                    feature_vectors[j].reshape(1, -1))[0]
        hausdorff_matrix[j, i] = hausdorff_matrix[i, j]

# Hausdorff距離を類似度に変換（ここでは1-Hausdorff距離）
#similarity_matrix = 1 - (hausdorff_matrix / np.max(hausdorff_matrix))


# コサイン類似度を計算
similarity_matrix = cosine_similarity(feature_vectors)

# グラフオブジェクトの作成
G = nx.Graph()

# ノード追加 (画像ファイル名)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# 上から順にME(271),KO(232),DY(223),PA(108),SU(116)の計950
labels = ['ME'] * 271 + ['KO'] * 232 + ['DY'] * 223 + ['PA'] * 108 + ['SU'] * 116

# ノードラベルを付与
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]


# k-NNモデルの作成と学習
k = 5  # 大きいと失敗する
knn = NearestNeighbors(n_neighbors=k + 1)  # 自分自身を含めるため+1
knn.fit(feature_vectors)

# 各ノードのk近傍を求める
distances, indices = knn.kneighbors(feature_vectors)

# エッジ追加 (k-NNに基づいて)
for i in range(len(feature_vectors)):
    for j in indices[i, 1:]:  # 自分自身を除外するため1:
        G.add_edge(nodes[i], nodes[j], weight=1 - distances[i, j])  # 距離を重みに変換

'''    
# エッジ追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.73 なら エッジ数: 1万以上. 0.74が最適
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
'''

def create_lle_graph(node_vectors, n_components, n_neighbors, metric='cosine'):
    """
    ノードベクトルからLLEを用いてグラフを生成する関数

    Args:
        node_vectors: 各ノードの634次元のベクトルを要素とするリスト
        n_components: LLEで埋め込む次元数
        n_neighbors: LLEで使用する近傍点の数
        metric: 類似度を計算する際のメトリクス

    Returns:
        NetworkXのグラフオブジェクト
    """

    # ノードベクトルをNumPy配列に変換
    X = np.array(node_vectors)

    # LLEによる次元削減
    embedding = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    X_transformed = embedding.fit_transform(X)

    # 低次元空間におけるノード間の距離に基づいてエッジを張る
    # 距離がthreshold以下のノード同士にエッジを張るようにする
    threshold = 0.389 # この値は小さすぎると計算に失敗するので注意.0.365で失敗
    G_lle = nx.Graph()
    for i in range(len(X_transformed)):
        for j in range(i+1, len(X_transformed)):
            distance = np.linalg.norm(X_transformed[i] - X_transformed[j])
            if distance <= threshold:
                G_lle.add_edge(i, j)

    return G_lle

# LLEで作ったグラフ
#G_lle = create_lle_graph(list(data.values()), 140, 15, 'cosine')
# 150で失敗か

num_edges = G.number_of_edges()
print("エッジ数:", num_edges)

# ノードラベル追加
for i, node in enumerate(G_lle.nodes()):
    G_lle.nodes[node]['label'] = labels[i]


def weighted_simrank(G, c=0.8, max_iter=100, eps=1e-6):
    """重み付き無向グラフに対してSimRankを計算する関数

    Args:
        G (nx.Graph): 重み付き無向グラフオブジェクト
        c (float): 減衰係数
        max_iter (int): 最大イテレーション数
        eps (float): 収束判定の閾値

    Returns:
        numpy.ndarray: SimRank行列
    """

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 隣接行列の作成 (重み付き)
    adj_matrix = nx.to_scipy_sparse_array(G, weight='weight')

    # 正規化
    degree_matrix_inv = sp.diags(1 / adj_matrix.sum(axis=1).flatten())
    transition_matrix = degree_matrix_inv @ adj_matrix

    # SimRankの初期化
    sim_matrix = sp.eye(n)

    for _ in range(max_iter):
        prev_sim_matrix = sim_matrix.copy()
        sim_matrix = c * transition_matrix.T @ (sim_matrix @ transition_matrix)
        sim_matrix.setdiag(1)

        # 収束判定
        diff = np.linalg.norm((sim_matrix - prev_sim_matrix).toarray(), ord='fro') # toarray() を使用
        if diff < eps:
            break

    return sim_matrix.toarray()

# SimRankの計算
#sim_matrix = weighted_simrank(G_lle)
sim_matrix = weighted_simrank(G)

# 結果の表示 (最初の5x5行列)
# print(sim_matrix[:5, :5])

'''
while True:
    try:
        user_input = int(input(f"ノード番号 (0-{len(nodes)-1}) を入力してください (終了は-1): "))
        if user_input == -1:
            break
        if 0 <= user_input < len(nodes):
            node_name = nodes[user_input]
            node_label = G.nodes[node_name]['label']
            sim_scores = list(enumerate(sim_matrix[user_input]))
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"{node_name} ({node_label}) と類似度の高いノード TOP 10 (SimRank):")
            simrank_labels = []
            for rank, (node_idx, score) in enumerate(sim_scores[:10]):
                neighbor_node_name = nodes[node_idx]
                neighbor_node_label = G.nodes[neighbor_node_name]['label']
                print(f"{rank+1}. {neighbor_node_name} ({neighbor_node_label}): {score:.4f}")
                simrank_labels.append(neighbor_node_label)

            same_label_count = simrank_labels.count(node_label)
            print(f"SimRankの結果のうち、入力ノードと同じラベルの数: {same_label_count}")

            # コサイン類似度によるTOP10の表示
            target_feature = feature_vectors[user_input]
            cosine_similarities = cosine_similarity([target_feature], feature_vectors)[0]
            cosine_sim_scores = list(enumerate(cosine_similarities))
            cosine_sim_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"{node_name} ({node_label}) と類似度の高いノード TOP 10 (Cosine Similarity):")
            cosine_labels = []
            for rank, (node_idx, score) in enumerate(cosine_sim_scores[1:11]): # 最初の要素は自分自身なのでスキップ
                neighbor_node_name = nodes[node_idx]
                neighbor_node_label = G.nodes[neighbor_node_name]['label']
                print(f"{rank+1}. {neighbor_node_name} ({neighbor_node_label}): {score:.4f}")
                cosine_labels.append(neighbor_node_label)

            same_label_count = cosine_labels.count(node_label)
            print(f"Cosine Similarityの結果のうち、入力ノードと同じラベルの数: {same_label_count}")

        else:
            print(f"無効なノード番号です。0-{len(nodes)-1} の範囲で入力してください。")
    except ValueError:
        print("整数を入力してください。")
'''


total_simrank_same_labels = 0
total_cosine_same_labels = 0

for user_input in range(len(nodes)):
    node_name = nodes[user_input]
    node_label = G.nodes[node_name]['label']
    sim_scores = list(enumerate(sim_matrix[user_input]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    simrank_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in sim_scores[:2]]
    same_label_count_simrank = simrank_labels.count(node_label)
    total_simrank_same_labels += same_label_count_simrank
    #print(f"ノード {user_input} ({node_name}, {node_label}) SimRank: 同じラベルの数 {same_label_count_simrank}")
    #target_feature = feature_vectors[user_input]
    #cosine_similarities = cosine_similarity([target_feature], feature_vectors)[0]
    #cosine_sim_scores = list(enumerate(cosine_similarities))
    #cosine_sim_scores.sort(key=lambda x: x[1], reverse=True)
    #cosine_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in cosine_sim_scores[1:3]]
    #same_label_count_cosine = cosine_labels.count(node_label)
    #total_cosine_same_labels += same_label_count_cosine
    #print(f"ノード {user_input} ({node_name}, {node_label}) Cosine Similarity: 同じラベルの数 {same_label_count_cosine}")
    # Hausdorff距離に基づく類似度スコアを計算
    hausdorff_similarities = 1 - (hausdorff_matrix[user_input] / np.max(hausdorff_matrix))
    hausdorff_sim_scores = list(enumerate(hausdorff_similarities))
    hausdorff_sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位2件のラベルを取得
    hausdorff_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in hausdorff_sim_scores[1:3]]
    
    # 同じラベルの数をカウント
    same_label_count_hausdorff = hausdorff_labels.count(node_label)
    total_cosine_same_labels += same_label_count_hausdorff

print(f"SimRank: 同じラベルの合計数 {(total_simrank_same_labels)/1900}, Cosine Similarity: 同じラベルの合計数 {(total_cosine_same_labels)/1900}")