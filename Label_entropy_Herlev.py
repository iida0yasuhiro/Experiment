# Herlev 917　でラベルエントロピーを評価する実験（本実験）
import json
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import mode
from collections import Counter

# HerlevのJSONファイル(917個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./H_full_merged_data.json', 'r') as f:
    data = json.load(f)

# 特徴ベクトルをNumPy配列に変換
feature_vectors = np.array(list(data.values()))

# コサイン類似度を計算
similarity_matrix = cosine_similarity(feature_vectors)


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


# グラフオブジェクトの作成
G = nx.Graph()

# ノードの追加 (画像ファイル名)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# ノードラベルのリストを作成
labels = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70

# ノードラベルを付与
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]

# ラベルが付与されたことを確認
#for node in G.nodes():
#    print(f"Node: {node}, Label: {G.nodes[node]['label']}")

# エッジの追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.73 なら エッジ数: 1万以上. 0.74が最適
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])


    maximal_cliques = []

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
    #print(len(clique), end=" ")  # ノード数表示
    #print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # ノードラベルを取得
        #print(f"{node}({label})", end="")  # ノード名とラベルを表示
        if i < len(clique) - 1:
            print(", ", end="")
    #print("}", end="  ")

    #print("ラベル数:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")

    # エントロピーを計算して表示
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"エントロピー: {entropy}")  # エントロピーを表示
    entropy_values.append(entropy) # リストに追加

# すべてのクリークのエントロピーの平均値を計算
average_entropy = np.mean(entropy_values)
print(f"全クリークのエントロピー平均値: {average_entropy}")



# グラフオブジェクトGから隣接行列を「S0」として行列に変換
# そのうえで、モデル学習をするため、数値データを0から1の間で非負の実数に変換して計算可能にしておく
S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
print(S)



# ここからは既知ラベルで埋めたY1（917 x 7）を作る。
# 最初から何行目かまでは同じパタンの行になるが、そのための各パターンの行数と対応する値のリスト
# 上から順にSD(197),NC(98),NS(74),CS(150),MD(146),LD(182),NI(70)の計917
'''
patterns = [
    (197, [1,0,0,0,0,0,0]),
    (98, [0,1,0,0,0,0,0]),
    (74, [0,0,1,0,0,0,0]),
    (150, [0,0,0,1,0,0,0]),
    (146, [0,0,0,0,1,0,0]),
    (182, [0,0,0,0,0,1,0]),
    (70, [0,0,0,0,0,0,1])
]
'''
patterns = [
    (197, [0.7,0,0,0.1,0.1,0.1,0]),
    (98, [0,1,0,0,0,0,0]),
    (74, [0,0,1,0,0,0,0]),
    (150, [0.1,0,0,0.7,0.1,0.1,0]),
    (146, [0.1,0,0,0.1,0.7,0.1,0]),
    (182, [0.1,0,0,0.1,0.1,0.7,0]),
    (70, [0,0,0,0,0,0,1])
]
'''
patterns = [
    (197, [0.7,0,0,0.1,0.1,0.1,0]),
    (98, [0,0.8,0.1,0,0,0,0.1]),
    (74, [0,0.1,0.8,0,0,0,0.1]),
    (150, [0.1,0,0,0.7,0.1,0.1,0]),
    (146, [0.1,0,0,0.1,0.7,0.1,0]),
    (182, [0.1,0,0,0.1,0.1,0.7,0]),
    (70, [0,0.1,0.1,0,0,0,0.8])
]
'''

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


# ここからは実験の準備として、正解である完全なる既知ラベル行列Y1から、わざと
# いくつかラベルを間違えた結果（ラベルノイズが入ったものを生成）の
# 行列を「Y0」とし、変更された行をchanged_rowとして出力しておく
Y0, changed_row = modify_matrix(Y1)
print(changed_row)





# ラベル行列Y0に従ってグラフGのノードラベルを更新
for i, node in enumerate(G.nodes()):
    # Fの各行はone-hotベクトルなので、最大値のインデックスがラベルに対応する
    predicted_label_index = np.argmax(Y0[i])
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



# 実験その1. 試行数。（ただし111回まで試行数を増加させても目立った変化はなし）
num_trials = 11

# 各試行の結果を格納するリスト（11個の行列Fを格納するリスト）
all_F = []

# ここから実験開始。初期ノードを選定するため、ランダムにゼロにした
# 初期ラベル行列Y2を作る（これを11回試行する。なお111回まで試しても優位な精度向上は見られなかった）

for _ in range(num_trials):
 # ★SG値。例えば　> 0.3 ということは全体の3割をゼロとして、7割をそのまま初期データとして残すということ
 # Y2は実験（ラベル伝播計算）のため便宜的に一時作成したもの
 Y2 = np.array([row if np.random.rand() > 0.2 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # ここからラベル伝播の式を計算。
 # ★SG値。0.010-0.020 の範囲が適する。
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
#print(f"F:{F}")
#print(f"F_final:{F_final}")

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
print(f"Fのエントロピー平均値: {average_entropy}")