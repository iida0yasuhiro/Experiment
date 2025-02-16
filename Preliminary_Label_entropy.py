# Herlev 917�@�Ń��x���G���g���s�[��]����������i���O���؁j

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import mode
from collections import Counter

# Herlev��JSON�t�@�C��(917��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
with open('./H_full_merged_data.json', 'r') as f:
    data = json.load(f)

# �����x�N�g����NumPy�z��ɕϊ�
feature_vectors = np.array(list(data.values()))

# �R�T�C���ގ��x���v�Z
similarity_matrix = cosine_similarity(feature_vectors)


def calculate_entropy(clique, labels):
    """
    �N���[�N���̃��x�����z����G���g���s�[���v�Z����֐�

    Args:
        clique: �N���[�N���̃m�[�h�̃��X�g
        labels: �e�m�[�h�̃��x�����i�[�������� (key: �m�[�h��, value: ���x��)

    Returns:
        �N���[�N�̃G���g���s�[�l
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


# �O���t�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# �m�[�h�̒ǉ� (�摜�t�@�C����)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# �m�[�h���x���̃��X�g���쐬
labels = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70

# �m�[�h���x����t�^
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]

# ���x�����t�^���ꂽ���Ƃ��m�F
#for node in G.nodes():
#    print(f"Node: {node}, Label: {G.nodes[node]['label']}")

# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.73 �Ȃ� �G�b�W��: 1���ȏ�. 0.74���œK
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])


    maximal_cliques = []

def find_maximal_cliques(graph):
    """
    �O���t���̋ɑ�N���[�N�����ׂČ�����֐� (�m�[�h��3�ȏ�A���̋ɑ�N���[�N�ɕ�܂������̂͏���)

    Args:
        graph: networkx�O���t�I�u�W�F�N�g

    Returns:
        �m�[�h��3�ȏ�ŁA���̋ɑ�N���[�N�ɕ�܂���Ȃ��ɑ�N���[�N�̃��X�g
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
                # ���̋ɑ�N���[�N�ɕ�܂���邩�m�F
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
            if len(R) >= 3:  # �m�[�h��3�ȏ�̃N���[�N�݂̂𒊏o
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

# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)

# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

#print("�ɑ�N���[�N (�m�[�h����������):")
for clique in sorted_cliques:
    # ���x�����J�E���g
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ���x�����z��\��
    print(len(clique), end=" ")  # �m�[�h���\��
    print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # �m�[�h���x�����擾
        #print(f"{node}({label})", end="")  # �m�[�h���ƃ��x����\��
        if i < len(clique) - 1:
            print(", ", end="")
    print("}", end="  ")

    print("���x����:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")
    print()  # ���s



'''
# �O���t�̉��� (�ύX�Ȃ�)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=30, font_size=5)

# �ő�N���[�N�������\�� (�ύX�Ȃ�)
nx.draw_networkx_nodes(G, pos, nodelist=maximum_clique, node_color='red', node_size=30)

plt.title("Graph and Clique")
plt.show()


print("�m�[�h��3�ȏ�̋ɑ�N���[�N:")
for clique in maximal_cliques:
    print(clique,len(clique))

# �O���t�̉��� (�ύX�Ȃ�)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15)

colors = ['red', 'green', 'blue', 'yellow', 'purple']
for i, clique in enumerate(maximal_cliques):
    nx.draw_networkx_nodes(graph, pos, nodelist=clique, node_color=colors[i % len(colors)], node_size=700)

plt.title("show graph")
plt.show()
'''