# すべてのカテゴリに同じ割合でノイズを入れた場合　5月7日実験
'''
# Package installation (hidden on docs website).
dependencies = ["sklearn", "matplotlib", "pandas", "networkx", "scipy"] # cleanlabを削除

if "google.colab" in str(get_ipython()):  # Check if it's running in Google Colab
    cmd = ' '.join([dep for dep in dependencies])
    %pip install $cmd
else:
    missing_dependencies = []
    for dependency in dependencies:
        try:
            __import__(dependency)
        except ImportError:
            missing_dependencies.append(dependency)

    if len(missing_dependencies) > 0:
        print("Missing required dependencies:")
        print(*missing_dependencies, sep=", ")
        print("\nPlease install them before running the rest of this notebook.")

%config InlineBackend.print_figure_kwargs={"facecolor": "w"}
'''
#5月7日にSIPaKMeDでグラフクリーク実験。ICAHS向けラベル伝播
#5月7日。初期ラベルをランダム選択する際に、数を調整できるようにした。
import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity # コサイン類似度計算のために使用
from collections import Counter # ラベル数カウントのために使用
from sklearn.preprocessing import normalize # ラベル伝播法のために使用
import networkx as nx # グラフ処理のために使用
import math # エントロピー計算のために使用
from scipy.stats import mode # ラベル伝播法のために使用
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score # 評価指標計算のために使用

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

# JSONファイル(950個のnodeをKey（ファイル名）-Value（特徴ベクトル）で格納)を読み込む
# ファイルパスは環境に合わせて適宜変更してください
try:
    with open('./S_merged_data.json', 'r') as f:
        data = json.load(f)
    nodes = list(data.keys())
    feature_vectors = np.array(list(data.values()))
    print(f"データファイル './S_merged_data.json' を読み込みました。ノード数: {len(nodes)}")

except FileNotFoundError:
    print("エラー: データファイル './S_merged_data.json' が見つかりません。ファイルパスを確認してください。")
    # 以降の処理は実行できないため、ここで終了
    exit()
except Exception as e:
    print(f"エラー: データファイル読み込み中にエラーが発生しました - {e}")
    # 以降の処理は実行できないため、ここで終了
    exit()


# コサイン類似度を計算
similarity_matrix = cosine_similarity(feature_vectors)
print("コサイン類似度行列を計算しました。")

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
        # labels辞書からノードのラベルを取得
        label = labels.get(node)
        if label is not None:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    entropy = 0
    num_nodes = len(clique)
    if num_nodes == 0: # クリークが空の場合はエントロピー0
        return 0

    for count in label_counts.values():
        probability = count / num_nodes
        if probability > 0: # log2(0)を避ける
            entropy -= probability * math.log2(probability)

    return entropy

def find_maximal_cliques(graph):
    """
    グラフ内の極大クリークをすべて見つける関数 (ノード数3つ以上、他の極大クリークに包含されるものは除く)
    ※ networkxの組み込み関数を使用し、ノード数3つ以上のフィルタリングを適用します。

    Args:
        graph: networkxグラフオブジェクト

    Returns:
        ノード数3つ以上で、他の極大クリークに包含されない極大クリークのリスト
    """

    maximal_cliques = []
    # networkxの組み込み関数を使用して極大クリークを効率的に見つけます
    for clique in nx.find_cliques(graph):
        if len(clique) >= 3: # ノード数3つ以上のクリークを対象
            maximal_cliques.append(set(clique)) # setに変換して包含関係の判定を容易にする

    # 他の極大クリークに包含されるクリークを除外
    filtered_cliques = []
    for c1 in maximal_cliques:
        is_subset = False
        for c2 in maximal_cliques:
            if c1 != c2 and c1.issubset(c2):
                is_subset = True
                break
        if not is_subset:
            filtered_cliques.append(list(c1)) # リストに戻す

    return filtered_cliques


# グラフオブジェクトの作成
G = nx.Graph()

# ノードの追加 (画像ファイル名)
G.add_nodes_from(nodes)

# ノードラベルのリストを作成 (SIPaKMeDのクラスと数に基づいています)
# これはノイズ注入前の「真のラベル」として使用します。
# 上から順にME(271),KO(232),DY(223),PA(108),SU(116)の計950

labels_list = ['ME'] * 271 + ['KO'] * 232 + ['DY'] * 223 + ['PA'] * 108 + ['SU'] * 116
# labels_list = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70

# Create a dictionary mapping node names to original labels
# This will be used as the "true label" before noise injection.
original_labels_dict = {nodes[i]: labels_list[i] for i in range(len(nodes))}

# Based on SIPaKMeD classes and counts (reiterated)
# In order from top: ME(271), KO(232), DY(223), PA(108), SU(116), totaling 950
# Calculate node counts per class
class_counts = Counter(original_labels_dict.values())
print("Node counts per class in the original data:", dict(class_counts))


# Noise Injection - Inject specified percentage of noise per category
noise_percentage = 0.1 # Noise percentage *** (Inject this percentage of noise into each category)
changed_nodes = [] # List of nodes whose labels will be changed

print(f"\nInjecting approximately {noise_percentage*100:.2f}% noise into each category.")

# Select nodes for noise injection for each class
for class_label, count in class_counts.items():
    nodes_in_class = [node for node, label in original_labels_dict.items() if label == class_label]
    num_noisy_in_class = int(count * noise_percentage)

    if num_noisy_in_class > 0 and num_noisy_in_class <= len(nodes_in_class):
        # Randomly select nodes for noise injection from within that class
        noisy_nodes_in_class = random.sample(nodes_in_class, num_noisy_in_class)
        changed_nodes.extend(noisy_nodes_in_class)
    elif num_noisy_in_class > len(nodes_in_class):
        print(f"Warning: Calculated number of noisy nodes ({num_noisy_in_class}) for class '{class_label}' exceeds the number of nodes in the class ({len(nodes_in_class)}). All nodes in this class will be set as noisy.")
        changed_nodes.extend(nodes_in_class)
    else:
        print(f"Info: No nodes to inject noise into for class '{class_label}' ({num_noisy_in_class} calculated).")


# Set the post-noise labels in the graph
# This label is used for clique detection and initial label selection, NOT directly as input for label spreading.
# The input for label spreading is only the initial labels selected from clique detection.
for node in G.nodes():
    if node in changed_nodes:
        original_label = original_labels_dict[node]
        # Randomly select a label different from the original label
        all_possible_labels = list(class_counts.keys()) # List of all unique labels
        possible_new_labels = [label for label in all_possible_labels if label != original_label]

        if possible_new_labels: # Only swap if there are possible new labels
            new_label = random.choice(possible_new_labels)
            G.nodes[node]['label'] = new_label # Save the post-noise label as a graph attribute
        else: # If there are no other label options (shouldn't happen, but for safety)
            G.nodes[node]['label'] = original_label # Do not change the label
    else:
        G.nodes[node]['label'] = original_labels_dict[node] # Set the original label

print(f"Total of {len(changed_nodes)} nodes selected for noise injection.")
# Count the number of nodes actually changed (result of random selection and swapping logic)
# Note: The changed_nodes list above is the list of "nodes selected as candidates for label change".
# To confirm if the label was actually changed (assigned a label different from the original), we count again.
actual_changed_count = sum(G.nodes[node]['label'] != original_labels_dict[node] for node in G.nodes())
print(f"Number of nodes whose labels were actually changed: {actual_changed_count}")



# エッジの追加 (類似度に基づいて)
# 類似度行列はすでに計算済み (similarity_matrix)
threshold = 0.74 # 類似度の閾値
edges_added = 0
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)): # 重複を避けるため j は i+1 から開始
        if similarity_matrix[i, j] > threshold:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
            edges_added += 1
print(f"類似度 {threshold} 以上のエッジを {edges_added} 個追加しました。")


# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)
print(f"ノード数3つ以上の極大クリークを {len(maximal_cliques)} 個見つけました。")


# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)
print(f"極大クリークをノード数でソートしました。")


# 各クリークの支配的なラベルを持つノードを初期ラベルとして抽出
initial_labels = {} # ラベル伝播の初期ラベルとして使用する辞書 {ノード名: ラベル}
clique_info = []
entropy_values = [] # クリークのエントロピー値を格納

print("\n各クリークの情報を処理中...")
for i, clique in enumerate(sorted_cliques):
    # クリーク内のノードの現在のラベルを取得 (ノイズ注入後のラベル)
    current_clique_labels = {node: G.nodes[node]['label'] for node in clique}

    # ラベル重みを計算 (クリーク内のエッジの重みの合計)
    label_weights = {}
    for node in clique:
        label = current_clique_labels[node]
        if label not in label_weights:
            label_weights[label] = 0
        for neighbor in G.neighbors(node):
            if neighbor in clique: # クリーク内の隣接ノードのみを考慮
                label_weights[label] += G[node][neighbor]['weight']

    # 支配的なラベルを決定 (ラベル重みが最大のラベル)
    dominant_label = None
    if label_weights:
        dominant_label = max(label_weights, key=label_weights.get)
    else: # ラベル重みが計算できない場合（例：エッジがないクリーク）
        # クリーク内のノードのラベルの最頻値を支配的なラベルとする
        clique_labels_list = list(current_clique_labels.values())
        if clique_labels_list:
             dominant_label_mode, _ = mode(clique_labels_list, keepdims=True) # keepdims=Trueで常に配列を返すように
             dominant_label = dominant_label_mode[0]
        else: # クリークが空（これは find_maximal_cliques で除外されるはずですが念のため）
             dominant_label = None # 支配的なラベルなし


    # ラベル数カウント
    label_counts = Counter(current_clique_labels.values())
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # エントロピーを計算してリストに追加
    entropy = calculate_entropy(clique, current_clique_labels)
    entropy_values.append(entropy)

    # 支配的なラベルを持つノードを初期ラベル候補として選択
    dominant_nodes_in_clique = [node for node in clique if current_clique_labels.get(node) == dominant_label]

    # 支配的なラベルを持つノードの中から、まだ初期ラベルとして選択されていないノードを優先的に選択
    initial_node = None
    for node in dominant_nodes_in_clique:
        if node not in initial_labels:
            initial_node = node
            break

    # 支配的なラベルを持つノードが全て初期ラベルとして選択済みの場合、
    # クリーク内の最初のノードを初期ラベルとして選択（これはあまり望ましくないかもしれません）
    if initial_node is None and dominant_nodes_in_clique:
         initial_node = dominant_nodes_in_clique[0]


    # 支配的なラベルが見つかり、かつ初期ノードが選択できた場合のみinitial_labelsに追加
    # この initial_labels がラベル伝播の「固定された」初期ラベルとして機能します。
    if dominant_label is not None and initial_node is not None:
         initial_labels[initial_node] = dominant_label


    # 追加: クリーク情報をリストに追加
    clique_info.append({
        "clique_id": i,
        "clique_size": len(clique),
        "dominant_label": dominant_label,
        "label_weights": label_weights,
        "label_counts": dict(sorted_label_counts), # Counterをdictに変換して保存
        "entropy": entropy,
        "initial_node_for_ls": initial_node # ラベル伝播に使う初期ノード (initial_labelsに含まれるノード)
    })

print(f"ラベル伝播の初期ラベルとして {len(initial_labels)} 個のノードを選択しました。")


def label_spreading(graph, initial_labels, alpha=0.1, max_iter=100, tol=1e-5):
    """ラベル伝播法を用いてラベルを修正する関数（★★★アルファ値を調整する）"""

    # initial_labelsからユニークなラベルのリストを作成
    unique_labels = sorted(list(set(initial_labels.values())))
    if not unique_labels:
        print("警告: initial_labelsが空です。ラベル伝播は実行されません。")
        # グラフの全ノードに対してNoneを返す辞書を作成
        return {node: None for node in graph.nodes()}


    label_indices = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    num_nodes = len(graph)
    nodes_list = list(graph.nodes()) # ノードのリストを作成してインデックスアクセス可能にする

    # 初期ラベル行列の作成 (F)
    # F[i, j] はノード i がラベル j を持つ確率の初期値
    F = np.zeros((num_nodes, num_labels))
    # 初期ラベルを持つノードに対応する行を1に設定
    # initial_labelsに含まれるノードは、そのラベルの確率が1、他のラベルの確率は0となる
    # initial_labelsに含まれないノードは、初期確率は全て0となる
    for node, label in initial_labels.items():
        if node in nodes_list: # グラフに存在するノードか確認
            node_idx = nodes_list.index(node)
            if label in label_indices: # 有効なラベルか確認
                 label_idx = label_indices[label]
                 F[node_idx, label_idx] = 1

    # グラフの隣接行列の作成 (W)
    # W[i, j] はノード i とノード j の間のエッジの重み
    W = nx.adjacency_matrix(graph, nodelist=nodes_list).toarray()

    # 正規化された隣接行列 (S) の計算
    # S = D^(-1/2) W D^(-1/2)
    # D は対角行列で、D[i, i] はノード i の次数（または重みの合計）
    row_sum = W.sum(axis=1)
    # ゼロ除算を防ぐ
    D_inv_sqrt = np.diag(np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0))
    S = np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)


    # ラベル伝播の反復計算
    # F(t+1) = alpha * S * F(t) + (1 - alpha) * Y
    # Y は初期ラベル行列 (初期値のみ1、その他0) - ラベルが固定される部分
    Y = np.copy(F) # 初期ラベル行列をYとして保持

    for iter_count in range(max_iter):
        F_new = alpha * np.dot(S, F) + (1 - alpha) * Y
        # 収束判定
        diff = np.linalg.norm(F_new - F)
        F = F_new
        if diff < tol:
            print(f"ラベル伝播が収束しました (iteration: {iter_count+1})")
            break
    else:
        print(f"警告: ラベル伝播が最大反復回数 ({max_iter}) に達しましたが収束しませんでした。")


    # 修正されたラベルの取得
    # 各ノードに対して、確率が最大となるラベルを選択
    # ただし、初期ラベルとして与えられたノードのラベルは固定されているはずなので、
    # そのノードのラベルは initial_labels の値を使用するのがより正確かもしれません。
    # ここでは伝播後のF行列の最大値を使用しますが、initial_labelsを優先するロジックも考えられます。
    new_labels_list = []
    for i in range(num_nodes):
        node = nodes_list[i]
        if node in initial_labels:
            # 初期ラベルとして与えられたノードは、そのラベルを最終ラベルとする
            new_labels_list.append(initial_labels[node])
        else:
            # それ以外のノードは、伝播後の確率が最大のラベルを選択
            if np.sum(F[i, :]) > 0: # 確率の合計が0より大きい場合のみラベルを決定
                 new_labels_list.append(unique_labels[np.argmax(F[i, :])])
            else:
                 # 確率が全て0の場合など、ラベルを決定できない場合はNoneとする
                 new_labels_list.append(None)


    # ノード名と修正されたラベルを対応させる辞書を作成
    new_labels_dict = {nodes_list[i]: new_labels_list[i] for i in range(num_nodes)}

    return new_labels_dict


# ラベル伝播法を用いてラベルを修正 (クリークの支配的なラベルを初期ラベルとして使用)
propagated_labels_dict = label_spreading(G, initial_labels)

# 評価の計算
# 評価は、ノイズ注入されたchanged_nodesに対して行います。
# ラベル伝播によって、これらのノードのラベルが元の正しいラベルにどれだけ戻ったかを評価します。

# 評価に使用する真のラベルは、ノイズ注入前の original_labels_dict です。
# 評価に使用する予測ラベルは、ラベル伝播後の propagated_labels_dict です。

# TP, TN, FP, FN を計算
# ここでの評価は「ノイズ注入されたノードのラベルが元のラベルに戻ったか」を基準とします。
# TP: ノイズ注入されたノードで、ラベル伝播後に元のラベルに戻ったもの
# FN: ノイズ注入されたノードで、ラベル伝播後も元のラベルに戻らなかったもの
# FP: ノイズ注入されていないノードで、ラベル伝播後に元のラベルから変わってしまったもの
# TN: ノイズ注入されていないノードで、ラベル伝播後も元のラベルのままだったもの

TP = 0 # True Positive: ノイズ注入され、伝播後に元のラベルに戻った
FN = 0 # False Negative: ノイズ注入され、伝播後も元のラベルに戻らなかった
FP = 0 # False Positive: ノイズ注入されていないが、伝播後にラベルが変わった
TN = 0 # True Negative: ノイズ注入されておらず、伝播後もラベルが変わらなかった

# changed_nodes (ノイズ注入されたノード) について評価
for node in changed_nodes:
    actual_original_label = original_labels_dict[node] # ノイズ注入前の真のラベル
    predicted_propagated_label = propagated_labels_dict.get(node) # ラベル伝播後のラベル

    if predicted_propagated_label is not None: # ラベル伝播でラベルが付与されたか確認
        if predicted_propagated_label == actual_original_label:
            TP += 1 # ノイズが修正された（元のラベルに戻った）
        else:
            FN += 1 # ノイズが修正されなかった（元のラベルに戻らなかった）
    # else: # ラベル伝播でラベルが付与されなかった場合（発生しにくいケースですが）
    #     FN += 1 # 元のラベルに戻らなかったとみなす


# changed_nodes 以外のノード (ノイズ注入されていないノード) について評価
# これらのノードの真のラベルは original_labels_dict に格納されています。
# ラベル伝播後もこのラベルのままであることが望ましいです。
unchanged_nodes = [node for node in nodes if node not in changed_nodes]

for node in unchanged_nodes:
    actual_original_label = original_labels_dict[node] # ノイズ注入前の真のラベル
    predicted_propagated_label = propagated_labels_dict.get(node) # ラベル伝播後のラベル

    if predicted_propagated_label is not None: # ラベル伝播でラベルが付与されたか確認
        if predicted_propagated_label == actual_original_label:
            TN += 1 # ラベルが元のまま維持された
        else:
            FP += 1 # ラベルが変わってしまった（誤って変更された）
    # else: # ラベル伝播でラベルが付与されなかった場合
    #     # 真のラベルが分からない、または評価対象外とするなど、状況に応じて判断が必要
    #     pass # ここでは評価に含めない


# 結果を表示
print("\nラベル伝播によるノイズ修正の評価 (クリークの支配的ラベルを初期ラベルとして使用):")
print("TP (ノイズ修正成功):", TP)
print("FN (ノイズ修正失敗):", FN)
print("FP (誤ってラベル変更):", FP)
print("TN (正しいラベル維持):", TN)

total_evaluated_nodes = TP + FN + FP + TN
print('合計評価ノード数：', total_evaluated_nodes)
print('全ノード数：', len(nodes))
# 注：total_evaluated_nodes は、ラベル伝播後にラベルが付与されたノードのうち、
# changed_nodes と unchanged_nodes に含まれるノードの合計です。
# ラベル伝播でラベルが付与されなかったノードは評価に含まれていません。


# 適合率、再現率、精度、F値を計算
# ここでの適合率と再現率は、「ノイズ注入されたノードをどれだけ元のラベルに戻せたか」
# という観点での評価指標となります。
# 適合率 (Precision): 修正されたと予測したノード（TP+FP）のうち、実際にノイズが修正された（TP）割合
# 再現率 (Recall): 実際にノイズ注入されたノード（TP+FN）のうち、ノイズが修正された（TP）割合
# 精度 (Accuracy): 全評価ノードのうち、正しく評価されたノード（TP+TN）の割合

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# 結果を表示
print("\nラベル伝播によるノイズ修正の評価指標 (クリークの支配的ラベルを初期ラベルとして使用):")
print(f"適合率 (Precision): {precision:.4f}")
print(f"再現率 (Recall): {recall:.4f}")
print(f"精度 (Accuracy): {accuracy:.4f}")
print(f"F値 (F1-score): {f1:.4f}")


# ランダムな初期ラベルを使用した場合の評価
# 新たにランダムな初期ラベルセットを作成し、再度ラベル伝播を実行して評価します。
print("\n--- ランダムな初期ラベルを使用した場合の評価 ---")

random_initial_labels = {}



'''
# 支配的なラベルの初期ラベル数と同じ数だけ、ランダムにノードを選択
num_initial_nodes = len(initial_labels) # クリークから選ばれた初期ラベル数と同じ数
if len(nodes) >= num_initial_nodes:
    random_initial_nodes = random.sample(nodes, num_initial_nodes)
    # 選択されたノードに、そのノードのノイズ注入後のラベルを初期ラベルとして設定
    for node in random_initial_nodes:
        if node in G.nodes(): # グラフに存在するノードか確認
            random_initial_labels[node] = G.nodes[node]['label'] # ノイズ注入後のラベルを使用
else:
    print("警告: ノード総数より初期ラベル数の方が多いため、ランダムな初期ラベルの選択をスキップします。")
    random_initial_labels = {} # 空の辞書を設定
'''

# ランダムな初期ラベルを使用した場合の評価
# 新たにランダムな初期ラベルセットを作成し、再度ラベル伝播を実行して評価します。
print("\n--- ランダムな初期ラベルを使用した場合の評価 ---")

random_initial_labels = {}
# 初期ラベルのパーセンテージを設定
random_initial_percentage = 0.95 # ★★★例: 10%のノードをランダムに選択

# ランダムに選択する初期ラベルの数を計算
num_random_initial_nodes = int(len(nodes) * random_initial_percentage)

# 選択されたノードに、そのノードのノイズ注入後のラベルを初期ラベルとして設定
if len(nodes) >= num_random_initial_nodes:
    random_initial_nodes = random.sample(nodes, num_random_initial_nodes)
    for node in random_initial_nodes:
        if node in G.nodes(): # グラフに存在するノードか確認
            random_initial_labels[node] = G.nodes[node]['label'] # ノイズ注入後のラベルを使用
else:
    print("警告: ノード総数より初期ラベル数の方が多いため、ランダムな初期ラベルの選択をスキップします。")
    random_initial_labels = {} # 空の辞書を設定



if random_initial_labels:
    print(f"ランダムな初期ラベルとして {len(random_initial_labels)} 個のノードを選択しました。")

    # ランダムな初期ラベルでラベル伝播を実行
    random_propagated_labels_dict = label_spreading(G, random_initial_labels)

    # ランダムな初期ラベルを使用した場合の TP, TN, FP, FN を計算
    random_TP = 0
    random_FN = 0
    random_FP = 0
    random_TN = 0

    # changed_nodes (ノイズ注入されたノード) について評価
    for node in changed_nodes:
        actual_original_label = original_labels_dict[node] # ノイズ注入前の真のラベル
        predicted_propagated_label = random_propagated_labels_dict.get(node) # ランダム初期ラベルでの伝播後のラベル

        if predicted_propagated_label is not None:
            if predicted_propagated_label == actual_original_label:
                random_TP += 1
            else:
                random_FN += 1
        # else:
        #     random_FN += 1

    # changed_nodes 以外のノード (ノイズ注入されていないノード) について評価
    for node in unchanged_nodes:
        actual_original_label = original_labels_dict[node] # ノイズ注入前の真のラベル
        predicted_propagated_label = random_propagated_labels_dict.get(node) # ランダム初期ラベルでの伝播後のラベル

        if predicted_propagated_label is not None:
            if predicted_propagated_label == actual_original_label:
                random_TN += 1
            else:
                random_FP += 1
        # else:
        #     pass


    print("\nランダムな初期ラベルでのノイズ修正の評価:")
    print("TP (ノイズ修正成功):", random_TP)
    print("FN (ノイズ修正失敗):", random_FN)
    print("FP (誤ってラベル変更):", random_FP)
    print("TN (正しいラベル維持):", random_TN)

    random_total_evaluated_nodes = random_TP + random_FN + random_FP + random_TN
    print('合計評価ノード数：', random_total_evaluated_nodes)


    # 適合率、再現率、精度、F値を計算（ランダムな初期ラベルを使用）
    random_precision = random_TP / (random_TP + random_FP) if (random_TP + random_FP) > 0 else 0
    random_recall = random_TP / (random_TP + random_FN) if (random_TP + random_FN) > 0 else 0
    random_accuracy = (random_TP + random_TN) / (random_TP + random_TN + random_FP + random_FN) if (random_TP + random_TN + random_FP + random_FN) > 0 else 0
    random_f1 = 2 * (random_precision * random_recall) / (random_precision + random_recall) if (random_precision + random_recall) > 0 else 0

    print("\nランダムな初期ラベルでのノイズ修正の評価指標:")
    print(f"適合率 (Precision): {random_precision:.4f}")
    print(f"再現率 (Recall): {random_recall:.4f}")
    print(f"精度 (Accuracy): {random_accuracy:.4f}")
    print(f"F値 (F1-score): {random_f1:.4f}")

else:
    print("\nランダムな初期ラベルが選択されなかったため、評価はスキップされます。")


# (オプション) 可視化
# 特徴量が2次元の場合のみ可視化
# 特徴空間での可視化はラベル伝播の過程を直接示すものではないため、ここではスキップします。
# 必要であれば、ノードの色を元のラベル、ノイズ注入後のラベル、伝播後のラベルで塗り分けるなどの可視化を追加できます。
# 例：nx.draw(G, with_labels=True, node_color=[G.nodes[n]['propagated_label'] for n in G.nodes()]) など

