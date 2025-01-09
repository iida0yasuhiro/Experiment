# Herlevのデータによる本実験。

import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import mode

# HerlevのJSONファイル(634個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./H_full_merged_data.json', 'r') as f:
    data = json.load(f)

# 特徴ベクトルをNumPy配列に変換
feature_vectors = np.array(list(data.values()))

# コサイン類似度を計算
similarity_matrix = cosine_similarity(feature_vectors)


# グラフオブジェクトの作成
G = nx.Graph()

# ノードの追加 (画像ファイル名)
nodes = list(data.keys())
G.add_nodes_from(nodes)


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
    threshold = 0.4 # この値は小さすぎると計算に失敗するので注意
    G_lle = nx.Graph()
    for i in range(len(X_transformed)):
        for j in range(i+1, len(X_transformed)):
            distance = np.linalg.norm(X_transformed[i] - X_transformed[j])
            if distance <= threshold:
                G_lle.add_edge(i, j)

    return G_lle

# LLEで作ったグラフ
# G_lle = create_lle_graph(list(data.values()), 100, 15, 'cosine')


# エッジの追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.73 なら エッジ数: 1万以上
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
#num_edges = G_lle.number_of_edges()
print("エッジ数:", num_edges)


'''
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# Print the detected communities
for i, community in enumerate(communities):
    print(f"Community {i+1}: {list(community)}")

# Calculate and print the modularity of the found communities
modularity_value = nx.algorithms.community.modularity(G, communities)
print(f"Modularity: {modularity_value}")
'''

degrees = dict(G.degree())
#degrees = dict(G_lle.degree())

# 次数の合計を計算
sum_of_degrees = sum(degrees.values())

# ノード数を取得
num_nodes = G.number_of_nodes()

# 平均次数を計算
average_degree = sum_of_degrees / num_nodes
print("平均次数:", average_degree)

isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# ネットワーク径
#diameter = nx.diameter(G)
#print("ネットワーク径:", diameter)

# 平均ノード間距離
#average_shortest_path_length = nx.average_shortest_path_length(G)
#print("平均ノード間距離:", average_shortest_path_length)

# クラスタ係数（グラフ全体の平均）
average_clustering = nx.average_clustering(G)
print("クラスタ係数:", average_clustering)

#average_clustering = nx.average_clustering(G_lle)
#print("クラスタ係数:", average_clustering)


# 各ノードのクラスタ係数
# node_clustering = nx.clustering(G)
# print("各ノードのクラスタ係数:", node_clustering)

'''
# x軸: 度数 (logスケール)
x = np.arange(min(degree_sequence), max(degree_sequence) + 1)
x = np.log10(x)  # 対数変換

# y軸: 度数を持つノード数 (logスケール)
y = [list(degree_sequence).count(i) for i in 10**x]  # xに対応する値でカウント
y = np.log10(y)

# 線形回帰モデルを作成
model = LinearRegression()

# データをreshapeしてモデルにフィット
X = x.reshape(-1, 1)
model.fit(X, y)

# 予測値
y_pred = model.predict(X)

# プロット
plt.loglog(10**x, 10**y, 'bo', label='data')
plt.loglog(10**x, 10**y_pred, 'r-', label='fit')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution')
plt.legend()
plt.grid(True)
plt.show()

# 回帰係数と切片
print('傾き:', model.coef_[0])
print('切片:', model.intercept_)
'''

'''
# 各ノードの次数をリストに格納
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# degree_sequence = sorted([d for n, d in G_lle.degree()], reverse=True)

# x軸: 度数 (logスケール)
x = np.arange(min(degree_sequence), max(degree_sequence) + 1)

# y軸: 度数を持つノード数 (logスケール)
y = [list(degree_sequence).count(i) for i in x]


plt.loglog(x, y, 'bo')  # log-logプロット
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Herlev Graph Degree Distribution')
plt.grid(True)
plt.show()
'''

'''
# 次数分布をヒストグラムで可視化
plt.hist(degree_sequence, bins=20, color='blue')
plt.title("Herlev Graph Degree Distribution")
plt.ylabel("# of node")
plt.xlabel("degree")
plt.show()
'''


'''
# データの分類境界
boundaries = [0, 128, 231, 300, 405, 456, 505, 643]
labels = ["light_dysplastic", "moderate_dysplastic", "normal_columnar", "carcinoma_in_situ", "normal_superficie", "normal_intermediate", "severe_dysplastic"]

# エッジの追加 (類似度に基づいて、かつ同じカテゴリ内のみでいったん試してみたもの。本実験では使わない)
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):  # i+1から始めることで重複を回避
        # カテゴリの判定
        category_i = -1
        category_j = -1
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= i < boundaries[k + 1]:
                category_i = k
            if boundaries[k] <= j < boundaries[k + 1]:
                category_j = k

        # 同じカテゴリ内かつ類似度が閾値以上の場合にエッジを追加
        if category_i == category_j and similarity_matrix[i, j] > 0.75:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
'''


# グラフオブジェクトGから隣接行列を「S0」として行列に変換
# そのうえで、モデル学習をするため、数値データを0から1の間で非負の実数に変換して計算可能にしておく
S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
print(S)

# G_lle
#S0 = nx.adjacency_matrix(G_lle)
#S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)


# ここからは既知ラベルで埋めたY1（917 x 7）を作る。
# 最初から何行目かまでは同じパタンの行になるが、そのための各パターンの行数と対応する値のリスト
# 上から順にSD(197),NC(98),NS(74),CS(150),MD(146),LD(182),NI(70)の計917
patterns = [
    (197, [1,0,0,0,0,0,0]),
    (98, [0,1,0,0,0,0,0]),
    (74, [0,0,1,0,0,0,0]),
    (150, [0,0,0,1,0,0,0]),
    (146, [0,0,0,0,1,0,0]),
    (182, [0,0,0,0,0,1,0]),
    (70, [0,0,0,0,0,0,1])
]

# 空のリストを作成しておく
matrix_list = []

# 各パターンで繰り返し、行を生成したリストを作る
for num_rows, row_values in patterns:
    # 指定された行数分の行を生成し、リストに追加
    matrix_list.extend([row_values] * num_rows)

# 作ったリストを、NumPy配列に変換
# 既知ラベル行列(正解値)「Y1」を作る（すべてのノードにラベルが付与されている）
Y1 = np.array(matrix_list)

# ラベルノイズを疑似的に生成する関数
def modify_matrix(matrix):
    """
    行列の10%の行をランダムに選択し、特定の要素を置換する関数
    Args:
        matrix: 処理対象の行列（Y1）
    Returns:
        処理後の行列, ランダムに選択された行のインデックスのリスト
    """

    # 行数を取得（917行）
    num_rows = matrix.shape[0]

    # ★SG値。ランダムノイズの割合をランダムに選択　0.1なら全体の10%に誤りをいれる
    random_indices = np.random.choice(num_rows, int(num_rows * 0.1), replace=False)

    # 選択された行に対して処理
    for i in random_indices:
        # 1を0に置換
        matrix[i, matrix[i] == 1] = 0

        # 残りの要素からランダムに1つを選択して1にする
        zero_indices = np.where(matrix[i] == 0)[0]
        random_zero_idx = np.random.choice(zero_indices)
        matrix[i, random_zero_idx] = 1

    return matrix, random_indices
# ラベルノイズ疑似生成関数ここまで

'''
def show_matching_rows(Y0, F, changed_row):
    """
    要素が同じ行の数をカウントし、一致した行のインデックスを返す関数

    Args:
        Y0: numpy.ndarray
        F: numpy.ndarray
        changed_row: int (未使用)

    Returns:
        list: 一致した行のインデックスのリスト
    """

    matching_indices = []
    for i in range(Y0.shape[0]):
        if np.array_equal(Y0[i], F[i]):
            matching_indices.append(i)
    return matching_indices
'''


# ここからは実験の準備として、正解である完全なる既知ラベル行列Y1から、わざと
# いくつかラベルを間違えた結果（ラベルノイズが入ったものを生成）の
# 行列を「Y0」とし、変更された行をchanged_rowとして出力しておく
Y0, changed_row = modify_matrix(Y1)

print(changed_row)

# 実験その1. 試行数。
num_trials = 11

# 各試行の結果を格納するリスト（11個の行列Fを格納するリスト）
all_F = []

# ここから実験開始。初期ノードを選定するため、ランダムにゼロにした
# 初期ラベル行列Y2を作る（これを11回試行する。なお111回まで試しても優位な精度向上は見られなかった）

for _ in range(num_trials):
 # ★SG値。例えば　> 0.3 ということは全体の3割をゼロとして、7割をそのまま初期データとして残すということ
 # Y2は実験（ラベル伝播計算）のため便宜的に一時作成したもの
 Y2 = np.array([row if np.random.rand() > 0.3 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # ここからラベル伝播の式を計算。
 # ★SG値。Set alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(917) #次元に合わせて調整
 inv_term = np.linalg.inv(I - alpha * S)
 F0 = inv_term.dot(Y2)
 #print(F0)

 # F0の各要素を0か1に統一する関数
 def process_matrix(matrix):
    # 負の値を0に
    matrix[matrix < 0] = 0
    # 各行の最大値のインデックスを取得
    max_indices = np.argmax(matrix, axis=1)
    # 最大値の要素を1、それ以外を0
    matrix = np.zeros_like(matrix)
    matrix[np.arange(matrix.shape[0]), max_indices] = 1
    return matrix


 # 最後に要素が0か１に統一された推定ラベル行列を最終的に得る
 F = process_matrix(F0)
 #print(F)
 all_F.append(F)

 # 以上で計算終わり。

F_final = np.array([mode(arr)[0] for arr in zip(*all_F)])
print(F_final)

# ところで、この計算結果であるFが完全なる既知ラベル行列Y1とどれだけ違ったかを検証してみたい
mask = Y1 != F_final


# マスク内のTrueの数をカウント（つまり、要素が異なる個数）
num_diff = np.count_nonzero(mask)


# これにより、提案方法が、もとのラベルをどこまで再現（ラベル伝播向けの良いグラフ）出来ていたかを検証する
# 教師が7割なら7.4パーセント(～6.2パーセント)の誤差！！ 教師が3割なら11.5パーセント
# 要素の数は 634 x 7 = 4438
print("正しい既知ラベル行列との差分がどれだけあるか:", num_diff)



# 最後は集計。入れたノイズ（意図的にわざと誤ったchanged_row）のうち、
# どれだけのラベルを当てられたか（変更が入ったか）

def count_matching_rows(matrix1, matrix2, row_indices):
    """
    指定した行番号の行の要素の並びが完全に一致している数をカウントする
    Args:
        matrix1, matrix2: 比較する2つの行列
        row_indices: 比較する行のインデックスのリスト(すなわちchanged_rowの範囲で比較)
    Returns:
        完全に一致した行の数
    """

    # 指定された行のみを取り出す
    selected_rows1 = matrix1[row_indices]
    selected_rows2 = matrix2[row_indices]

    # 各行の比較結果をTrue/Falseで表す
    comparison_result = (selected_rows1 == selected_rows2).all(axis=1)

    # Trueの数をカウント
    matching_count = np.sum(comparison_result)

    return matching_count

# そしてこれが、この実験でもっとも行いたかった結果。Channged_rowの範囲でふたつの
# 行列の行が一致した、ということは、計算結果Fが、Y0で混入されたラベルノイズを検出できたということ
matching_count = count_matching_rows(Y0, F, changed_row)
print("matched:")
print(matching_count)


#matching_indices = show_matching_rows(Y0, F, 0)
#print("matched rows at indices:")
#print(matching_indices)


def find_different_rows(Y0, F):
    """
    要素が異なる行のインデックスを返す関数

    Args:
        Y0: numpy.ndarray
        F: numpy.ndarray

    Returns:
        list: 要素が異なる行のインデックスのリスト
    """

    different_indices = []
    for i in range(Y0.shape[0]):
        if not np.array_equal(Y0[i], F[i]):
            different_indices.append(i)
    return different_indices


different_indices = find_different_rows(Y1, F)
#print("要素が異なる行のインデックス:", different_indices)

set1 = set(changed_row)
set2 = set(different_indices)

#共通要素、すなわち、我々の実験で検出に失敗したノードのリスト（行番号）
common_elements = set1.intersection(set2)
print("共通要素:", list(common_elements))

#changed_row（の集合型）set1から共通要素（の集合型）common_elementsを差し引いた集合が、実験で検出に成功したノードとなる。
set_success = set1 - common_elements
print("成功", list(set_success))

# ここからは考察のため、検出に失敗したノードのPageRankと、検出できたノードのPageRankを比較する
pr = nx.pagerank(G,alpha=0.75,weight='weight')

# ノードIDと行番号の対応を辞書に格納
node_to_index = {node: index for index, node in enumerate(G.nodes)}

# 検出に失敗したlist内の行番号に対応するPageRankスコアを抽出
result1 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in common_elements}

# スコアのリストを作成
scores = [score for node, (score, _) in result1.items()]

# スコアの平均値を計算
average_score = sum(scores) / len(scores)

print("検出失敗のPRスコアの平均値:", average_score)
#print(result1)

# 逆に、検出に成功したlist内の行番号に対応するPageRankスコアを抽出
result2 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in set_success}

# スコアのリストを作成
scores = [score for node, (score, _) in result2.items()]

# スコアの平均値を計算
average_score = sum(scores) / len(scores)

print("検出成功のPRスコアの平均値:", average_score)
#print(result2)

'''
# スプリングレイアウト
pos = nx.spring_layout(G, k=0.85, iterations=130, weight='weight')
# pos = nx.circular_layout(G)

# ノードとエッジの描画
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=0.1)

# ラベル表示を調整
# ノード数が多いため、ラベル表示は省略または一部のみ表示する
for node, (x, y) in pos.items():
     if node in nodes[:10]:  # 例：先頭10個のノードのみラベルを表示
         plt.text(x, y, node, fontsize=4, ha='center', va='center')

plt.axis('off')
plt.show()
'''