# SipakMedの分析　ラベルエントロピー（本実験）

import json
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from scipy import stats
from scipy.stats import mode
from collections import Counter

# JSONファイル(950個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./S_merged_data.json', 'r') as f:
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

# ノードラベルのリストを作成
# 上から順にME(271),KO(232),DY(223),PA(108),SU(116)の計950
labels = ['ME'] * 271 + ['KO'] * 232 + ['DY'] * 223 + ['PA'] * 108 + ['SU'] * 116

# ノードラベルを付与
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]

# エッジの追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.75 なら エッジ数: xx
        if similarity_matrix[i, j] > 0.73:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("エッジ数:", num_edges)

maximal_cliques = []

def calculate_entropy(clique, labels):
    """
    クリーク内のラベル分布からエントロピーを計算する関数

    Args:
        clique: クリーク内のノードのリスト
        labels: 各ノードのラベルを格納した辞書 (key: ノード名, value: ラベル)

    Returns:
        クリークのエントロピー値
    """

    label_counts = {}
    for node in clique:
        label = labels[node]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    entropy = 0
    num_nodes = len(clique)
    for count in label_counts.values():
        probability = count / num_nodes
        entropy -= probability * math.log2(probability)

    return entropy

def find_maximal_cliques(graph):
    """
    グラフ内の極大クリークをすべて見つける関数 (ノード数3つ以上、他の極大クリークに包含されるものは除く)

    Args:
        graph: networkxグラフオブジェクト

    Returns:
        ノード数3つ以上で、他の極大クリークに包含されない極大クリークのリスト
    """

    maximal_cliques = []
    for node in graph.nodes():
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

        if len(clique) >= 10:
            is_maximal = True
            for existing_clique in maximal_cliques:
                if clique.issubset(existing_clique):
                    is_maximal = False
                    break

            if is_maximal:
                # 他の極大クリークに包含されるか確認
                is_subset = False
                for other_clique in maximal_cliques:
                    if clique != other_clique and clique.issubset(other_clique):
                        is_subset = True
                        break

                if not is_subset:
                    maximal_cliques.append(clique)

    return maximal_cliques



# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)

# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values = [] # エントロピー値を格納するリスト

#print("極大クリーク (ノード数が多い順):")
for clique in sorted_cliques:
    # ラベル数カウント
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ラベル分布を表示
    print(len(clique), end=" ")  # ノード数表示
    print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # ノードラベルを取得
        #print(f"{node}({label})", end="")  # ノード名とラベルを表示
        if i < len(clique) - 1:
            print(", ", end="")
    print("}", end="  ")

    print("ラベル数:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")

    # エントロピーを計算して表示
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"エントロピー: {entropy}")  # エントロピーを表示
    entropy_values.append(entropy) # リストに追加

# すべてのクリークのエントロピーの平均値を計算
average_entropy = np.mean(entropy_values)
print(f"全クリークのエントロピー平均値: {average_entropy}")

# 次数の合計を計算
#sum_of_degrees = sum(degrees.values())

# ノード数を取得
num_nodes = G.number_of_nodes()

# 平均次数を計算
#average_degree = sum_of_degrees / num_nodes
#print("平均次数:", average_degree)

# クラスタ係数（グラフ全体の平均）
#average_clustering = nx.average_clustering(G)
#print("クラスタ係数:", average_clustering)


# 各ノードの次数をリストに格納
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# x軸: 度数 (logスケール)
x = np.arange(min(degree_sequence), max(degree_sequence) + 1)

# y軸: 度数を持つノード数 (logスケール)
y = [list(degree_sequence).count(i) for i in x]


'''
# データの分類境界
boundaries = [0, 128, 231, 300, 405, 456, 505, 643]
labels = ["light_dysplastic", "moderate_dysplastic", "normal_columnar", "carcinoma_in_situ", "normal_superficie", "normal_intermediate", "severe_dysplastic"]
'''

# グラフオブジェクトGから隣接行列を「S0」として行列に変換
# そのうえで、モデル学習をするため、数値データを0から1の間で非負の実数に変換して計算可能にしておく
S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)


# ここからは既知ラベルで埋めたY1（950 x 5）を作る。
# 最初から何行目かまでは同じパタンの行になるが、そのための各パターンの行数と対応する値のリスト
# 上から順にME(271),KO(232),DY(223),PA(108),SU(116)の計950

patterns = [
    (271, [0.8,0.1,0.1,0,0]),
    (232, [0.1,0.8,0.1,0,0]),
    (223, [0.1,0.1,0.8,0,0]),
    (108, [0,0,0,1,0]),
    (116, [0,0,0,0,1])
]
'''
patterns = [
    (271, [1,0,0,0,0]),
    (232, [0,1,0,0,0]),
    (223, [0,0,1,0,0]),
    (108, [0,0,0,1,0]),
    (116, [0,0,0,0,1])
]
'''
# 空のリストを作成しておく
matrix_list = []

# 各パターンで繰り返し、行を生成したリストを作る
for num_rows, row_values in patterns:
    # 指定された行数分の行を生成し、リストに追加
    matrix_list.extend([row_values] * num_rows)

# 作ったリストを、NumPy配列に変換
# 既知ラベル行列「Y1」を作る（すべてのノードにラベルが付与されている）
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

    # 行数を取得（950行）
    num_rows = matrix.shape[0]

    # ★SG値。ランダムノイズの割合をランダムに選択　0.1なら全体の10%に誤りをいれる
    random_indices = np.random.choice(num_rows, int(num_rows * 0.05), replace=False)

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


# ここからは実験の準備として、完全なる既知ラベル行列Y1から、わざと
# いくつかラベルを間違えた結果（ラベルノイズが入ったものを生成）の
# 行列を「Y0」とし、変更された行をchanged_rowとして出力しておく
Y0, changed_row = modify_matrix(Y1)
print(changed_row)


# ラベル行列Y0に従ってグラフGのノードラベルを更新
for i, node in enumerate(G.nodes()):
    # Fの各行はone-hotベクトルなので、最大値のインデックスがラベルに対応する
    predicted_label_index = np.argmax(Y0[i])
    # ラベルリストlabelsから対応するラベルを取得
    predicted_label = ['ME', 'KO', 'DY', 'PA', 'SU'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # ノードラベルを更新

# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)

# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # エントロピー値を格納するリスト

# 更新されたラベル分布をクリークごとに出力
for clique in sorted_cliques:
    # ラベル数カウント
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ラベル分布を表示
    #print(len(clique), end=" ")  # ノード数表示
    #print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # 更新されたノードラベルを取得
        if i < len(clique) - 1:
            print(", ", end="")
    #print("}", end="  ")

    #print("ラベル数:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # エントロピーを計算して表示
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"エントロピー: {entropy}")  # エントロピーを表示
    entropy_values_after.append(entropy) # リストに追加

# すべてのクリークのエントロピーの平均値を計算
average_entropy = np.mean(entropy_values_after)
print(f"Y0のエントロピー平均値: {average_entropy}")





# 実験その1. 試行数。
num_trials = 11
# 各試行の結果を格納するリスト（11個の行列Fを格納するリスト）
all_F = []

# ここから実験開始。初期ノードを選定するため、ランダムにゼロにした初期ノード
# ラベル行列Y2を作る（これを11回試行するwhileの修正をここから加える）

for _ in range(num_trials):
 # ★SG値。例えば　> 0.3 ということは全体の3割をゼロとして、7割をそのまま初期データとして残すということ
 Y2 = np.array([row if np.random.rand() > 0.5 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # ここからラベル伝播の式を計算。
 # ★SG値。Set alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(950)
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


# ラベル行列Y0に従ってグラフGのノードラベルを更新
for i, node in enumerate(G.nodes()):
    # Fの各行はone-hotベクトルなので、最大値のインデックスがラベルに対応する
    predicted_label_index = np.argmax(F[i])
    # ラベルリストlabelsから対応するラベルを取得
    predicted_label = ['ME', 'KO', 'DY', 'PA', 'SU'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # ノードラベルを更新

# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)

# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # エントロピー値を格納するリスト

# 更新されたラベル分布をクリークごとに出力
for clique in sorted_cliques:
    # ラベル数カウント
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ラベル分布を表示
    #print(len(clique), end=" ")  # ノード数表示
    #print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # 更新されたノードラベルを取得
        if i < len(clique) - 1:
            print(", ", end="")
    #print("}", end="  ")

    #print("ラベル数:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # エントロピーを計算して表示
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"エントロピー: {entropy}")  # エントロピーを表示
    entropy_values_after.append(entropy) # リストに追加

# すべてのクリークのエントロピーの平均値を計算
average_entropy = np.mean(entropy_values_after)
print(f"Fのエントロピー平均値: {average_entropy}")




F_final = np.array([mode(arr)[0] for arr in zip(*all_F)])
#print(F_final)

# ところで、この計算結果であるFが完全なる既知ラベル行列Y1とどれだけ違ったかを検証してみたい
mask = Y1 != F_final


# マスク内のTrueの数をカウント（つまり、要素が異なる個数）
num_diff = np.count_nonzero(mask)


# これにより、提案方法が、もとのラベルをどこまで再現（ラベル伝播向けの良いグラフ）出来ていたかを検証する
# 教師が7割なら7.4パーセント(～6.2パーセント)の誤差！！ 教師が3割なら11.5パーセント
# 要素の数は 950 x 5 = 4750
print("正しい既知ラベル行列との差分がどれだけあるか:", num_diff)



# 最後は集計。入れたノイズのうち、どれだけのラベルを当てられたか（変更が入ったか）

def count_matching_rows(matrix1, matrix2, row_indices):
    """
    指定した行番号の行の要素の並びが完全に一致している数をカウントする
    Args:
        matrix1, matrix2: 比較する2つの行列
        row_indices: 比較する行のインデックスのリスト
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



variances = []
for row_idx in changed_row:
    row_data = F0[row_idx, :]  # 指定行のデータを取得
    variance = np.var(row_data)
    variances.append(variance)



print("検出成功したlabel scoreの分散", variances)

mean_variance = np.mean(variances)

print("検出成功したlabel scoreの分散の平均値", mean_variance)


unchanged_row = list(common_elements)

variances2 = []
for row_idx in unchanged_row:
    row_data = F0[row_idx, :]  # 指定行のデータを取得
    variance = np.var(row_data)
    variances2.append(variance)


print("検出失敗したlabel scoreの分散", variances2)

mean_variance2 = np.mean(variances2)

print("検出失敗したlabel scoreの分散の平均値", mean_variance2)


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

print("検出失敗のPRスコアの平均値:", round(average_score,6))
#print(result1)

# 逆に、検出に成功したlist内の行番号に対応するPageRankスコアを抽出
result2 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in set_success}

# スコアのリストを作成
scores = [score for node, (score, _) in result2.items()]

# スコアの平均値を計算
average_score = sum(scores) / len(scores)

print("検出成功のPRスコアの平均値:", round(average_score,6))
#print(result2)


'''
# ラベル行列Fに従ってグラフGのノードラベルを更新
for i, node in enumerate(G.nodes()):
    # Fの各行はone-hotベクトルなので、最大値のインデックスがラベルに対応する
    predicted_label_index = np.argmax(F[i])
    # ラベルリストlabelsから対応するラベルを取得
    predicted_label = ['SD', 'NC', 'NS', 'CS', 'MD', 'LD', 'NI'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # ノードラベルを更新

# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)

# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # エントロピー値を格納するリスト

# 更新されたラベル分布をクリークごとに出力
for clique in sorted_cliques:
    # ラベル数カウント
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ラベル分布を表示
    print(len(clique), end=" ")  # ノード数表示
    print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # 更新されたノードラベルを取得
        if i < len(clique) - 1:
            print(", ", end="")
    print("}", end="  ")

    print("ラベル数:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # エントロピーを計算して表示
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"エントロピー: {entropy}")  # エントロピーを表示
    entropy_values_after.append(entropy) # リストに追加

# すべてのクリークのエントロピーの平均値を計算
average_entropy = np.mean(entropy_values_after)
print(f"全クリークのエントロピー平均値: {average_entropy}")
'''
