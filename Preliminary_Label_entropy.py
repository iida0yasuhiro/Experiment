# Herlev 917　でラベルエントロピーを評価する実験（事前検証）

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

'''
def bron_kerbosch(graph):
    def _bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 3:  # ノード数3つ以上のクリークのみを抽出
                yield R
            return

        v = P.pop()
        for node in list(P):
            if node in graph.neighbors(v):
                yield from _bron_kerbosch(R | {v}, P & set(graph.neighbors(v)), X & set(graph.neighbors(v)))
            P.remove(node)
            X.add(node)

    for start_node in graph.nodes():
        yield from _bron_kerbosch({start_node}, set(graph.neighbors(start_node)), set())

def find_maximal_cliques(graph):
    maximal_cliques = []
    for clique in bron_kerbosch(graph):
        is_maximal = True
        for other_clique in maximal_cliques:
            if clique.issubset(other_clique):
                is_maximal = False
                break
        if is_maximal:
            maximal_cliques.append(clique)
    return maximal_cliques
'''

# 極大クリークを見つける
maximal_cliques = find_maximal_cliques(G)

# 極大クリークをノード数が多い順にソート
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

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
    print()  # 改行



'''
# グラフの可視化 (変更なし)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=30, font_size=5)

# 最大クリークを強調表示 (変更なし)
nx.draw_networkx_nodes(G, pos, nodelist=maximum_clique, node_color='red', node_size=30)

plt.title("Graph and Clique")
plt.show()


print("ノード数3つ以上の極大クリーク:")
for clique in maximal_cliques:
    print(clique,len(clique))

# グラフの可視化 (変更なし)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15)

colors = ['red', 'green', 'blue', 'yellow', 'purple']
for i, clique in enumerate(maximal_cliques):
    nx.draw_networkx_nodes(graph, pos, nodelist=clique, node_color=colors[i % len(colors)], node_size=700)

plt.title("show graph")
plt.show()
'''