# Build Graph from Herlev

import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from scipy import stats
from scipy.stats import mode

# HerlevのJSONファイル(634個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./merged_data.json', 'r') as f:
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

# エッジの追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.75 なら エッジ数: 4362
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("エッジ数:", num_edges)

'''
# 各ノードの次数をリストに格納
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# 次数分布をヒストグラムで可視化
plt.hist(degree_sequence, bins=20, color='blue')
plt.title("degree distribution")
plt.ylabel("# of node")
plt.xlabel("degree")
plt.show()
'''


'''
# データの分類境界
boundaries = [0, 128, 231, 300, 405, 456, 505, 643]
labels = ["light_dysplastic", "moderate_dysplastic", "normal_columnar", "carcinoma_in_situ", "normal_superficie", "normal_intermediate", "severe_dysplastic"]

# エッジの追加 (類似度に基づいて、かつ同じカテゴリ内のみ)
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
# print(S)


# ここからは既知ラベルで埋めたY1（634 x 7）を作る。
# 最初から何行目かまでは同じパタンの行になるが、そのための各パターンの行数と対応する値のリスト
# 上から順にLD(128),MD(103),NC(69),CS(105),NS(51),NI(49),SD(138)の計634
patterns = [
    (128, [1,0,0,0,0,0,0]),
    (103, [0,1,0,0,0,0,0]),
    (69, [0,0,1,0,0,0,0]),
    (105, [0,0,0,1,0,0,0]),
    (51, [0,0,0,0,1,0,0]),
    (49, [0,0,0,0,0,1,0]),
    (138, [0,0,0,0,0,0,1])
]

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

    # 行数を取得（634行）
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


# ここからは実験の準備として、完全なる既知ラベル行列Y1から、わざと
# いくつかラベルを間違えた結果（ラベルノイズが入ったものを生成）の
# 行列を「Y0」とし、変更された行をchanged_rowとして出力しておく
Y0, changed_row = modify_matrix(Y1)

print(changed_row)

# 実験その1. 試行数。
num_trials = 11
# 各試行の結果を格納するリスト（11個の行列Fを格納するリスト）
all_F = []

# ここから実験開始。初期ノードを選定するため、ランダムにゼロにした初期ノード
# ラベル行列Y2を作る（これを11回試行するwhileの修正をここから加える）

for _ in range(num_trials):
 # ★SG値。例えば　> 0.3 ということは全体の3割をゼロとして、7割をそのまま初期データとして残すということ
 Y2 = np.array([row if np.random.rand() > 0.3 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # ここからラベル伝播の式を計算。
 # ★SG値。Set alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(643)
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