# 6月8日。DSAA向けLabel commonization 実験開始 - Herlev

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
from builtins import min

# HerlevのJSONファイル(917個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
with open('./HFA_merged.json', 'r') as f:
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

'''
# LLEで作ったグラフ
G_lle = create_lle_graph(list(data.values()), 100, 15, 'cosine')
'''

# エッジの追加 (類似度に基づいて)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ★エッジ類似度の閾値を調整　0.75 なら エッジ数: 4362
        if similarity_matrix[i, j] > 0.75:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("エッジ数:", num_edges)

degrees = dict(G.degree())
#degrees = dict(G_lle.degree())

# 次数の合計を計算
sum_of_degrees = sum(degrees.values())


# ノード数を取得
num_nodes = G.number_of_nodes()

'''
# 平均次数を計算
average_degree = sum_of_degrees / num_nodes
print("平均次数:", average_degree)

isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
'''

# 各ノードの次数をリストに格納
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# x軸: 度数 (logスケール)
#x = np.arange(min(degree_sequence), max(degree_sequence) + 1)

# y軸: 度数を持つノード数 (logスケール)
#y = [list(degree_sequence).count(i) for i in x]


# グラフオブジェクトGから隣接行列を「S0」として行列に変換
# そのうえで、モデル学習をするため、数値データを0から1の間で非負の実数に変換して計算可能にしておく
#S0 = nx.adjacency_matrix(G)
#S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)
# G_lle

S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array

# ここからは既知ラベルで埋めたY1（917 x 7）を作る。
# 最初から何行目かまでは同じパタンの行になるが、そのための各パターンの行数と対応する値のリスト
# 上から順にLD(182),MD(146),CS(150),SD(197),NC(98),NS(74),NI(70)の計917
# Abnormal 675, Normal 242
patterns = [
    (182, [1,0,0,0,0,0,0]),
    (146, [0,1,0,0,0,0,0]),
    (150, [0,0,1,0,0,0,0]),
    (197, [0,0,0,1,0,0,0]),
    (98, [0,0,0,0,1,0,0]),
    (74, [0,0,0,0,0,1,0]),
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


def noise_input(matrix, max_row=675):
    """
    行列の1行目からmax_row行目までの範囲で、10%の行をランダムに選択し、特定の要素を置換する関数
    Args:
        matrix: 処理対象の行列（Y1）
        max_row: ノイズを導入する最大行数 (デフォルト: 675)
    Returns:
        処理後の行列, ランダムに選択された行のインデックスのリスト
    """

    # 指定された行数までの範囲で処理
    num_rows = min(max_row, matrix.shape[0])

    # ノイズ混入★★★　ランダムに選択する行のインデックスをSG値で設定。0.1なら10%混入
    random_indices = np.random.choice(num_rows, int(num_rows * 0.3), replace=False)

    # 選択された行に対して処理
    for i in random_indices:
        # 1を0に置換
        matrix[i, matrix[i] == 1] = 0

        # 残りの要素からランダムに1つを選択して1にする
        zero_indices = np.where(matrix[i] == 0)[0]
        random_zero_idx = np.random.choice(zero_indices)
        matrix[i, random_zero_idx] = 1

    return matrix, random_indices

# ここからは実験の準備として、正解である完全なる既知ラベル行列Y1から、わざと
# いくつかラベルを間違えた結果（ラベルノイズが入ったものを生成）の
# 行列を「Y0」とし、変更された行をchanged_rowとして出力しておく
Y0, changed_row = noise_input(Y1)

changed_row.sort()
print(changed_row)
print("ノイズ要素数",len(changed_row))

##### 7分類を6分類に縮退させる関数####
def reduce_matrix(Y):
    """
    行列Yの6列目と7列目を縮約する関数
    Args:
        Y: 917行7列のnumpy.ndarray
    Returns:
        Z: 917行6列のnumpy.ndarray
    """

    # 6列目と7列目を抜き出し、どちらか一方が1であれば1、そうでなければ0にする
    reduced_col = np.any(Y[:, 5:7], axis=1).astype(int)

    # 元の行列Yから6列目を削除し、新しい列を追加
    Z = np.delete(Y, 6, axis=1)
    Z = np.column_stack((Z, reduced_col.reshape(-1, 1)))

    return Z

# ラベルノイズの入ったY0に対して、縮退関数を呼び出してZを取得　6列に縮退している
#Z = reduce_matrix(Y0)
#print(Z)


##### 7分類を5分類に縮退させる関数####
def reduce_matrix_modified(Y):
    """
    行列Yの左端から5列目、6列目、7列目を縮約する関数
    Args:
        Y: 917行7列のnumpy.ndarray
    Returns:
        Z_5: 917行5列のnumpy.ndarray
    """

    # 5列目、6列目、7列目を抜き出し、どれか一つでも1があれば1、そうでなければ0にする
    reduced_col = np.any(Y[:, 4:7], axis=1).astype(int)

    # 元の行列Yから5列目、6列目、7列目を削除し、新しい列を追加
    Z_5 = np.delete(Y, [4, 5, 6], axis=1)
    Z_5 = np.column_stack((Z_5, reduced_col.reshape(-1, 1)))

    return Z_5

Z_5 = reduce_matrix_modified(Y0)
#print(Z_5)

# 実験その1. 試行数。
num_trials = 11

#### ここからY0（7列の完全版）のラベル伝播　ただし教師無しのためアンサンプル

# 各試行の結果を格納するリスト（11個の行列Fを格納するリスト）
all_F = []

#### ここからラベル伝播（5列に縮退したZ_5に対するもの）　ただし教師無しのためアンサンプル
all_F_red = []

# ここから実験開始。初期ノードを選定するため、ランダムにゼロにした
# 初期ラベル行列Y2を作る（これを11回試行する。なお111回まで試しても優位な精度向上は見られなかった）

for _ in range(num_trials):

 # 行番号ごとの閾値設定
 thresholds[:675] = 0.2  # 異常カテゴリ４分類:675個。1行目から675行目までを0.2に変更　これは80%の初期ラベルを使うということ★★★
 thresholds[675:917] = 0.8  # 正常カテゴリ1分類：242個。676行目から917行目までを0.4に変更.これは60%の初期ラベルを使うということ　★★★

 # 行列Yの作成
 Y3 = np.array([row if np.random.rand() > thresholds[i] else np.zeros_like(row) for i, row in enumerate(Z_5)])

 # ここからラベル伝播の式を計算。
 # ★SG値。Set alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(917)
 inv_term = np.linalg.inv(I - alpha * S)
 F1 = inv_term.dot(Y3)

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
 F_red = process_matrix(F1) # 5列に縮退したもの
 #print(F)
 all_F_red.append(F_red)

F_final_red = np.array([mode(arr)[0] for arr in zip(*all_F_red)])
#print(F_final_red)
#### ここまでの塊がラベル伝播

Y5 = Y0[:, :5] # Y0はノイズを混入した7列の完全版。これを左から5列だけ切り出したのがY5
#F5 = F_final[:, :5]  # 7列の完全版の計算結果がF_final。これを左から5列だけ切り出したのがF5
#F5_red = F_final_red[:, :5] # 6列に縮退したときの計算結果がF_final_red。これを左から5列だけ切り出したのがF5_red

#print("Y5", Y5.shape[0], Y5.shape[1])
#print("F_final_red", F_final_red.shape[0], F_final_red.shape[1])


def compare_rows(F, Y):
  return np.where(~(F == Y).all(axis=1))[0]

diff_rows = compare_rows(F_final_red, Y5)
print("不一致の行番号:", diff_rows)

def extract_numbers_less_than_675_np(numbers):
  """
  NumPy配列から675以下の数値のみを抽出する関数
  Args:
    numbers: NumPy配列
  Returns:
    675以下の数値のNumPy配列
  """
  return numbers[numbers <= 675]

ext_num = extract_numbers_less_than_675_np(diff_rows)
print(ext_num)
print("検出した要素数",len(ext_num))

common_elements = set(changed_row) & set(diff_rows)
list_common = list(common_elements)
list_common.sort()
print(list_common)
print("どちらにも含まれる要素数",len(common_elements))


variances = []
for row_idx in common_elements:
    row_data = F1[row_idx, :]  # 指定行のデータを取得
    variance = np.var(row_data)
    variances.append(variance)

mean_variance = np.mean(variances)
print("検出成功したlabel scoreの分散の平均値", mean_variance)

fail_rows = list(set(changed_row) - set(common_elements))

for row_idx in fail_rows:
    row_data = F1[row_idx, :]  # 指定行のデータを取得
    variance = np.var(row_data)
    variances.append(variance)

mean_variance = np.mean(variances)
print("検出失敗したlabel scoreの分散の平均値", mean_variance)

def calculate_precision_recall(diff_rows, changed_rows):
    """
    適合率と再現率を計算する関数
    Args:
        diff_rows (list): モデルが誤りと予測した要素のリスト
        changed_rows (list): 実際に誤っていた要素のリスト
    Returns:
        tuple: 適合率と再現率のタプル
    """
    # 真陽性 (True Positive): モデルが誤りと予測し、実際に誤っていた要素
    true_positives = set(diff_rows).intersection(set(changed_rows))
    # 偽陽性 (False Positive): モデルが誤りと予測したが、実際には正しい要素
    false_positives = set(diff_rows) - true_positives
    # 偽陰性 (False Negative): モデルが正しく予測したが、実際には誤っていた要素
    false_negatives = set(changed_rows) - true_positives
    # 適合率 (Precision) = 真陽性 / (真陽性 + 偽陽性)
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    # 再現率 (Recall) = 真陽性 / (真陽性 + 偽陰性)
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0

    return precision, recall

precision, recall = calculate_precision_recall(ext_num, changed_row)
print("Precision:", precision)
print("再現率:", recall)

# ここから追加するコード

print("\n--- ラベル修正精度の評価 ---")

# 1. 真の正解ラベル（ノイズ導入前のY1）を5列に縮退させる
# Y1はノイズ導入前の完全な正解ラベル（7列）
Y1_red_for_eval = reduce_matrix_modified(Y1)

# 2. ラベル伝播後の最終予測結果 F_final_red （5列）と比較

# 全体のラベル修正精度
correctly_corrected_all = np.sum((F_final_red == Y1_red_for_eval).all(axis=1))
total_rows = Y1_red_for_eval.shape[0]
overall_correction_accuracy = correctly_corrected_all / total_rows
print(f"全体のラベル修正精度: {overall_correction_accuracy:.4f}")

# ノイズが導入された行（changed_row）におけるラベル修正精度
# F_final_redとY1_red_for_evalのchanged_rowの部分を比較
corrected_noisy_rows = 0
for idx in changed_row:
    # ラベル伝播の結果 (F_final_red[idx]) が、元の正しいラベル (Y1_red_for_eval[idx]) と一致するか
    if np.array_equal(F_final_red[idx], Y1_red_for_eval[idx]):
        corrected_noisy_rows += 1

total_noisy_rows = len(changed_row)
noise_correction_accuracy = corrected_noisy_rows / total_noisy_rows if total_noisy_rows > 0 else 0

print(f"ノイズが導入された行におけるラベル修正精度: {noise_correction_accuracy:.4f}")

# 正常カテゴリ（675行目以降）におけるラベル修正精度（ノイズが導入されていない部分）
# Z_5の作成時に正常カテゴリの閾値は0.4なので、この部分にも初期ラベルがゼロになった箇所がある
normal_category_start_idx = 675
corrected_normal_rows = 0
total_normal_rows = 0
for idx in range(normal_category_start_idx, Y1_red_for_eval.shape[0]):
    total_normal_rows += 1
    if np.array_equal(F_final_red[idx], Y1_red_for_eval[idx]):
        corrected_normal_rows += 1

normal_correction_accuracy = corrected_normal_rows / total_normal_rows if total_normal_rows > 0 else 0
print(f"正常カテゴリ（ノイズなし）におけるラベル修正精度: {normal_correction_accuracy:.4f}")