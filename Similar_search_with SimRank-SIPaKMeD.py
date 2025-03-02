# Sipakmed�@��ǂݍ����SimRank���v�Z

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

# JSON�t�@�C��(950��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
with open('./S_merged_data.json', 'r') as f:
    data = json.load(f)

# �����x�N�g����NumPy�z��ɕϊ�
feature_vectors = np.array(list(data.values()))


# Hausdorff�����s����v�Z
num_images = feature_vectors.shape[0]
hausdorff_matrix = np.zeros((num_images, num_images))
for i in range(num_images):
    for j in range(i + 1):
        # �Ώ̍s��Ȃ̂ŁA��O�p�s��̂݌v�Z
        # Reshape feature vectors to 2D arrays for directed_hausdorff
        hausdorff_matrix[i, j] = directed_hausdorff(feature_vectors[i].reshape(1, -1),
                                                    feature_vectors[j].reshape(1, -1))[0]
        hausdorff_matrix[j, i] = hausdorff_matrix[i, j]

# Hausdorff������ގ��x�ɕϊ��i�����ł�1-Hausdorff�����j
#similarity_matrix = 1 - (hausdorff_matrix / np.max(hausdorff_matrix))


# �R�T�C���ގ��x���v�Z
similarity_matrix = cosine_similarity(feature_vectors)

# �O���t�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# �m�[�h�ǉ� (�摜�t�@�C����)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# �ォ�珇��ME(271),KO(232),DY(223),PA(108),SU(116)�̌v950
labels = ['ME'] * 271 + ['KO'] * 232 + ['DY'] * 223 + ['PA'] * 108 + ['SU'] * 116

# �m�[�h���x����t�^
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]


# k-NN���f���̍쐬�Ɗw�K
k = 5  # �傫���Ǝ��s����
knn = NearestNeighbors(n_neighbors=k + 1)  # �������g���܂߂邽��+1
knn.fit(feature_vectors)

# �e�m�[�h��k�ߖT�����߂�
distances, indices = knn.kneighbors(feature_vectors)

# �G�b�W�ǉ� (k-NN�Ɋ�Â���)
for i in range(len(feature_vectors)):
    for j in indices[i, 1:]:  # �������g�����O���邽��1:
        G.add_edge(nodes[i], nodes[j], weight=1 - distances[i, j])  # �������d�݂ɕϊ�

'''    
# �G�b�W�ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.73 �Ȃ� �G�b�W��: 1���ȏ�. 0.74���œK
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
'''

def create_lle_graph(node_vectors, n_components, n_neighbors, metric='cosine'):
    """
    �m�[�h�x�N�g������LLE��p���ăO���t�𐶐�����֐�

    Args:
        node_vectors: �e�m�[�h��634�����̃x�N�g����v�f�Ƃ��郊�X�g
        n_components: LLE�Ŗ��ߍ��ގ�����
        n_neighbors: LLE�Ŏg�p����ߖT�_�̐�
        metric: �ގ��x���v�Z����ۂ̃��g���N�X

    Returns:
        NetworkX�̃O���t�I�u�W�F�N�g
    """

    # �m�[�h�x�N�g����NumPy�z��ɕϊ�
    X = np.array(node_vectors)

    # LLE�ɂ�鎟���팸
    embedding = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    X_transformed = embedding.fit_transform(X)

    # �᎟����Ԃɂ�����m�[�h�Ԃ̋����Ɋ�Â��ăG�b�W�𒣂�
    # ������threshold�ȉ��̃m�[�h���m�ɃG�b�W�𒣂�悤�ɂ���
    threshold = 0.389 # ���̒l�͏���������ƌv�Z�Ɏ��s����̂Œ���.0.365�Ŏ��s
    G_lle = nx.Graph()
    for i in range(len(X_transformed)):
        for j in range(i+1, len(X_transformed)):
            distance = np.linalg.norm(X_transformed[i] - X_transformed[j])
            if distance <= threshold:
                G_lle.add_edge(i, j)

    return G_lle

# LLE�ō�����O���t
#G_lle = create_lle_graph(list(data.values()), 140, 15, 'cosine')
# 150�Ŏ��s��

num_edges = G.number_of_edges()
print("�G�b�W��:", num_edges)

# �m�[�h���x���ǉ�
for i, node in enumerate(G_lle.nodes()):
    G_lle.nodes[node]['label'] = labels[i]


def weighted_simrank(G, c=0.8, max_iter=100, eps=1e-6):
    """�d�ݕt�������O���t�ɑ΂���SimRank���v�Z����֐�

    Args:
        G (nx.Graph): �d�ݕt�������O���t�I�u�W�F�N�g
        c (float): �����W��
        max_iter (int): �ő�C�e���[�V������
        eps (float): ���������臒l

    Returns:
        numpy.ndarray: SimRank�s��
    """

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # �אڍs��̍쐬 (�d�ݕt��)
    adj_matrix = nx.to_scipy_sparse_array(G, weight='weight')

    # ���K��
    degree_matrix_inv = sp.diags(1 / adj_matrix.sum(axis=1).flatten())
    transition_matrix = degree_matrix_inv @ adj_matrix

    # SimRank�̏�����
    sim_matrix = sp.eye(n)

    for _ in range(max_iter):
        prev_sim_matrix = sim_matrix.copy()
        sim_matrix = c * transition_matrix.T @ (sim_matrix @ transition_matrix)
        sim_matrix.setdiag(1)

        # ��������
        diff = np.linalg.norm((sim_matrix - prev_sim_matrix).toarray(), ord='fro') # toarray() ���g�p
        if diff < eps:
            break

    return sim_matrix.toarray()

# SimRank�̌v�Z
#sim_matrix = weighted_simrank(G_lle)
sim_matrix = weighted_simrank(G)

# ���ʂ̕\�� (�ŏ���5x5�s��)
# print(sim_matrix[:5, :5])

'''
while True:
    try:
        user_input = int(input(f"�m�[�h�ԍ� (0-{len(nodes)-1}) ����͂��Ă������� (�I����-1): "))
        if user_input == -1:
            break
        if 0 <= user_input < len(nodes):
            node_name = nodes[user_input]
            node_label = G.nodes[node_name]['label']
            sim_scores = list(enumerate(sim_matrix[user_input]))
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"{node_name} ({node_label}) �Ɨގ��x�̍����m�[�h TOP 10 (SimRank):")
            simrank_labels = []
            for rank, (node_idx, score) in enumerate(sim_scores[:10]):
                neighbor_node_name = nodes[node_idx]
                neighbor_node_label = G.nodes[neighbor_node_name]['label']
                print(f"{rank+1}. {neighbor_node_name} ({neighbor_node_label}): {score:.4f}")
                simrank_labels.append(neighbor_node_label)

            same_label_count = simrank_labels.count(node_label)
            print(f"SimRank�̌��ʂ̂����A���̓m�[�h�Ɠ������x���̐�: {same_label_count}")

            # �R�T�C���ގ��x�ɂ��TOP10�̕\��
            target_feature = feature_vectors[user_input]
            cosine_similarities = cosine_similarity([target_feature], feature_vectors)[0]
            cosine_sim_scores = list(enumerate(cosine_similarities))
            cosine_sim_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"{node_name} ({node_label}) �Ɨގ��x�̍����m�[�h TOP 10 (Cosine Similarity):")
            cosine_labels = []
            for rank, (node_idx, score) in enumerate(cosine_sim_scores[1:11]): # �ŏ��̗v�f�͎������g�Ȃ̂ŃX�L�b�v
                neighbor_node_name = nodes[node_idx]
                neighbor_node_label = G.nodes[neighbor_node_name]['label']
                print(f"{rank+1}. {neighbor_node_name} ({neighbor_node_label}): {score:.4f}")
                cosine_labels.append(neighbor_node_label)

            same_label_count = cosine_labels.count(node_label)
            print(f"Cosine Similarity�̌��ʂ̂����A���̓m�[�h�Ɠ������x���̐�: {same_label_count}")

        else:
            print(f"�����ȃm�[�h�ԍ��ł��B0-{len(nodes)-1} �͈̔͂œ��͂��Ă��������B")
    except ValueError:
        print("��������͂��Ă��������B")
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
    #print(f"�m�[�h {user_input} ({node_name}, {node_label}) SimRank: �������x���̐� {same_label_count_simrank}")
    #target_feature = feature_vectors[user_input]
    #cosine_similarities = cosine_similarity([target_feature], feature_vectors)[0]
    #cosine_sim_scores = list(enumerate(cosine_similarities))
    #cosine_sim_scores.sort(key=lambda x: x[1], reverse=True)
    #cosine_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in cosine_sim_scores[1:3]]
    #same_label_count_cosine = cosine_labels.count(node_label)
    #total_cosine_same_labels += same_label_count_cosine
    #print(f"�m�[�h {user_input} ({node_name}, {node_label}) Cosine Similarity: �������x���̐� {same_label_count_cosine}")
    # Hausdorff�����Ɋ�Â��ގ��x�X�R�A���v�Z
    hausdorff_similarities = 1 - (hausdorff_matrix[user_input] / np.max(hausdorff_matrix))
    hausdorff_sim_scores = list(enumerate(hausdorff_similarities))
    hausdorff_sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # ���2���̃��x�����擾
    hausdorff_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in hausdorff_sim_scores[1:3]]
    
    # �������x���̐����J�E���g
    same_label_count_hausdorff = hausdorff_labels.count(node_label)
    total_cosine_same_labels += same_label_count_hausdorff

print(f"SimRank: �������x���̍��v�� {(total_simrank_same_labels)/1900}, Cosine Similarity: �������x���̍��v�� {(total_cosine_same_labels)/1900}")