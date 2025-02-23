# Herlev 917�@��ǂݍ����SimRank���v�Z

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

# �O���t�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# �m�[�h�ǉ� (�摜�t�@�C����)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# �m�[�h���x���̃��X�g���쐬
labels = ['SD'] * 197 + ['NC'] * 98 + ['NS'] * 74 + ['CS'] * 150 + ['MD'] * 146 + ['LD'] * 182 + ['NI'] * 70

# �m�[�h���x���ǉ�
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]

# �G�b�W�ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.73 �Ȃ� �G�b�W��: 1���ȏ�. 0.74���œK
        if similarity_matrix[i, j] > 0.76:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])



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
    threshold = 0.368 # ���̒l�͏���������ƌv�Z�Ɏ��s����̂Œ���.0.365�Ŏ��s
    G_lle = nx.Graph()
    for i in range(len(X_transformed)):
        for j in range(i+1, len(X_transformed)):
            distance = np.linalg.norm(X_transformed[i] - X_transformed[j])
            if distance <= threshold:
                G_lle.add_edge(i, j)

    return G_lle

# LLE�ō�����O���t
G_lle = create_lle_graph(list(data.values()), 100, 15, 'cosine')
# 150�Ŏ��s��

num_edges = G_lle.number_of_edges()
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
sim_matrix = weighted_simrank(G_lle)

# ���ʂ̕\�� (�ŏ���5x5�s��)
# print(sim_matrix[:5, :5])


total_simrank_same_labels = 0
total_cosine_same_labels = 0

for user_input in range(len(nodes)):
    node_name = nodes[user_input]
    node_label = G.nodes[node_name]['label']
    sim_scores = list(enumerate(sim_matrix[user_input]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    simrank_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in sim_scores[:3]]
    same_label_count_simrank = simrank_labels.count(node_label)
    total_simrank_same_labels += same_label_count_simrank
    print(f"�m�[�h {user_input} ({node_name}, {node_label}) SimRank: �������x���̐� {same_label_count_simrank}")

    target_feature = feature_vectors[user_input]
    cosine_similarities = cosine_similarity([target_feature], feature_vectors)[0]
    cosine_sim_scores = list(enumerate(cosine_similarities))
    cosine_sim_scores.sort(key=lambda x: x[1], reverse=True)
    cosine_labels = [G.nodes[nodes[node_idx]]['label'] for node_idx, _ in cosine_sim_scores[1:4]]
    same_label_count_cosine = cosine_labels.count(node_label)
    total_cosine_same_labels += same_label_count_cosine
    print(f"�m�[�h {user_input} ({node_name}, {node_label}) Cosine Similarity: �������x���̐� {same_label_count_cosine}")

print(f"SimRank: �������x���̍��v�� {total_simrank_same_labels}, Cosine Similarity: �������x���̍��v�� {total_cosine_same_labels}")