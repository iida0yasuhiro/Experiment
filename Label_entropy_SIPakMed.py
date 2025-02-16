# SipakMed�̕��́@���x���G���g���s�[�i�{�����j

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

# JSON�t�@�C��(950��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
with open('./S_merged_data.json', 'r') as f:
    data = json.load(f)

# �����x�N�g����NumPy�z��ɕϊ�
feature_vectors = np.array(list(data.values()))

# �R�T�C���ގ��x���v�Z
similarity_matrix = cosine_similarity(feature_vectors)


# �O���t�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# �m�[�h�̒ǉ� (�摜�t�@�C����)
nodes = list(data.keys())
G.add_nodes_from(nodes)

# �m�[�h���x���̃��X�g���쐬
# �ォ�珇��ME(271),KO(232),DY(223),PA(108),SU(116)�̌v950
labels = ['ME'] * 271 + ['KO'] * 232 + ['DY'] * 223 + ['PA'] * 108 + ['SU'] * 116

# �m�[�h���x����t�^
for i, node in enumerate(G.nodes()):
    G.nodes[node]['label'] = labels[i]

# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.75 �Ȃ� �G�b�W��: xx
        if similarity_matrix[i, j] > 0.73:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("�G�b�W��:", num_edges)

maximal_cliques = []

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



# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)

# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values = [] # �G���g���s�[�l���i�[���郊�X�g

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

    # �G���g���s�[���v�Z���ĕ\��
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"�G���g���s�[: {entropy}")  # �G���g���s�[��\��
    entropy_values.append(entropy) # ���X�g�ɒǉ�

# ���ׂẴN���[�N�̃G���g���s�[�̕��ϒl���v�Z
average_entropy = np.mean(entropy_values)
print(f"�S�N���[�N�̃G���g���s�[���ϒl: {average_entropy}")

# �����̍��v���v�Z
#sum_of_degrees = sum(degrees.values())

# �m�[�h�����擾
num_nodes = G.number_of_nodes()

# ���ώ������v�Z
#average_degree = sum_of_degrees / num_nodes
#print("���ώ���:", average_degree)

# �N���X�^�W���i�O���t�S�̂̕��ρj
#average_clustering = nx.average_clustering(G)
#print("�N���X�^�W��:", average_clustering)


# �e�m�[�h�̎��������X�g�Ɋi�[
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# x��: �x�� (log�X�P�[��)
x = np.arange(min(degree_sequence), max(degree_sequence) + 1)

# y��: �x�������m�[�h�� (log�X�P�[��)
y = [list(degree_sequence).count(i) for i in x]


'''
# �f�[�^�̕��ދ��E
boundaries = [0, 128, 231, 300, 405, 456, 505, 643]
labels = ["light_dysplastic", "moderate_dysplastic", "normal_columnar", "carcinoma_in_situ", "normal_superficie", "normal_intermediate", "severe_dysplastic"]
'''

# �O���t�I�u�W�F�N�gG����אڍs����uS0�v�Ƃ��čs��ɕϊ�
# ���̂����ŁA���f���w�K�����邽�߁A���l�f�[�^��0����1�̊ԂŔ񕉂̎����ɕϊ����Čv�Z�\�ɂ��Ă���
S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)


# ��������͊��m���x���Ŗ��߂�Y1�i950 x 5�j�����B
# �ŏ����牽�s�ڂ��܂ł͓����p�^���̍s�ɂȂ邪�A���̂��߂̊e�p�^�[���̍s���ƑΉ�����l�̃��X�g
# �ォ�珇��ME(271),KO(232),DY(223),PA(108),SU(116)�̌v950

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
# ��̃��X�g���쐬���Ă���
matrix_list = []

# �e�p�^�[���ŌJ��Ԃ��A�s�𐶐��������X�g�����
for num_rows, row_values in patterns:
    # �w�肳�ꂽ�s�����̍s�𐶐����A���X�g�ɒǉ�
    matrix_list.extend([row_values] * num_rows)

# ��������X�g���ANumPy�z��ɕϊ�
# ���m���x���s��uY1�v�����i���ׂẴm�[�h�Ƀ��x�����t�^����Ă���j
Y1 = np.array(matrix_list)

# ���x���m�C�Y���^���I�ɐ�������֐�
def modify_matrix(matrix):
    """
    �s���10%�̍s�������_���ɑI�����A����̗v�f��u������֐�
    Args:
        matrix: �����Ώۂ̍s��iY1�j
    Returns:
        ������̍s��, �����_���ɑI�����ꂽ�s�̃C���f�b�N�X�̃��X�g
    """

    # �s�����擾�i950�s�j
    num_rows = matrix.shape[0]

    # ��SG�l�B�����_���m�C�Y�̊����������_���ɑI���@0.1�Ȃ�S�̂�10%�Ɍ��������
    random_indices = np.random.choice(num_rows, int(num_rows * 0.05), replace=False)

    # �I�����ꂽ�s�ɑ΂��ď���
    for i in random_indices:
        # 1��0�ɒu��
        matrix[i, matrix[i] == 1] = 0

        # �c��̗v�f���烉���_����1��I������1�ɂ���
        zero_indices = np.where(matrix[i] == 0)[0]
        random_zero_idx = np.random.choice(zero_indices)
        matrix[i, random_zero_idx] = 1

    return matrix, random_indices
# ���x���m�C�Y�^�������֐������܂�


# ��������͎����̏����Ƃ��āA���S�Ȃ���m���x���s��Y1����A�킴��
# ���������x�����ԈႦ�����ʁi���x���m�C�Y�����������̂𐶐��j��
# �s����uY0�v�Ƃ��A�ύX���ꂽ�s��changed_row�Ƃ��ďo�͂��Ă���
Y0, changed_row = modify_matrix(Y1)
print(changed_row)


# ���x���s��Y0�ɏ]���ăO���tG�̃m�[�h���x�����X�V
for i, node in enumerate(G.nodes()):
    # F�̊e�s��one-hot�x�N�g���Ȃ̂ŁA�ő�l�̃C���f�b�N�X�����x���ɑΉ�����
    predicted_label_index = np.argmax(Y0[i])
    # ���x�����X�glabels����Ή����郉�x�����擾
    predicted_label = ['ME', 'KO', 'DY', 'PA', 'SU'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # �m�[�h���x�����X�V

# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)

# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # �G���g���s�[�l���i�[���郊�X�g

# �X�V���ꂽ���x�����z���N���[�N���Ƃɏo��
for clique in sorted_cliques:
    # ���x�����J�E���g
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ���x�����z��\��
    #print(len(clique), end=" ")  # �m�[�h���\��
    #print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # �X�V���ꂽ�m�[�h���x�����擾
        if i < len(clique) - 1:
            print(", ", end="")
    #print("}", end="  ")

    #print("���x����:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # �G���g���s�[���v�Z���ĕ\��
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"�G���g���s�[: {entropy}")  # �G���g���s�[��\��
    entropy_values_after.append(entropy) # ���X�g�ɒǉ�

# ���ׂẴN���[�N�̃G���g���s�[�̕��ϒl���v�Z
average_entropy = np.mean(entropy_values_after)
print(f"Y0�̃G���g���s�[���ϒl: {average_entropy}")





# ��������1. ���s���B
num_trials = 11
# �e���s�̌��ʂ��i�[���郊�X�g�i11�̍s��F���i�[���郊�X�g�j
all_F = []

# ������������J�n�B�����m�[�h��I�肷�邽�߁A�����_���Ƀ[���ɂ��������m�[�h
# ���x���s��Y2�����i�����11�񎎍s����while�̏C�����������������j

for _ in range(num_trials):
 # ��SG�l�B�Ⴆ�΁@> 0.3 �Ƃ������Ƃ͑S�̂�3�����[���Ƃ��āA7�������̂܂܏����f�[�^�Ƃ��Ďc���Ƃ�������
 Y2 = np.array([row if np.random.rand() > 0.5 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # �������烉�x���`�d�̎����v�Z�B
 # ��SG�l�BSet alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(950)
 inv_term = np.linalg.inv(I - alpha * S)
 F0 = inv_term.dot(Y2)
 #print(F0)

 # F0�̊e�v�f��0��1�ɓ��ꂷ��֐�
 def process_matrix(matrix):
    # ���̒l��0��
    matrix[matrix < 0] = 0
    # �e�s�̍ő�l�̃C���f�b�N�X���擾
    max_indices = np.argmax(matrix, axis=1)
    # �ő�l�̗v�f��1�A����ȊO��0
    matrix = np.zeros_like(matrix)
    matrix[np.arange(matrix.shape[0]), max_indices] = 1
    return matrix


 # �Ō�ɗv�f��0���P�ɓ��ꂳ�ꂽ���胉�x���s����ŏI�I�ɓ���
 F = process_matrix(F0)
 #print(F)
 all_F.append(F)

 # �ȏ�Ōv�Z�I���B


# ���x���s��Y0�ɏ]���ăO���tG�̃m�[�h���x�����X�V
for i, node in enumerate(G.nodes()):
    # F�̊e�s��one-hot�x�N�g���Ȃ̂ŁA�ő�l�̃C���f�b�N�X�����x���ɑΉ�����
    predicted_label_index = np.argmax(F[i])
    # ���x�����X�glabels����Ή����郉�x�����擾
    predicted_label = ['ME', 'KO', 'DY', 'PA', 'SU'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # �m�[�h���x�����X�V

# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)

# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # �G���g���s�[�l���i�[���郊�X�g

# �X�V���ꂽ���x�����z���N���[�N���Ƃɏo��
for clique in sorted_cliques:
    # ���x�����J�E���g
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ���x�����z��\��
    #print(len(clique), end=" ")  # �m�[�h���\��
    #print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # �X�V���ꂽ�m�[�h���x�����擾
        if i < len(clique) - 1:
            print(", ", end="")
    #print("}", end="  ")

    #print("���x����:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # �G���g���s�[���v�Z���ĕ\��
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"�G���g���s�[: {entropy}")  # �G���g���s�[��\��
    entropy_values_after.append(entropy) # ���X�g�ɒǉ�

# ���ׂẴN���[�N�̃G���g���s�[�̕��ϒl���v�Z
average_entropy = np.mean(entropy_values_after)
print(f"F�̃G���g���s�[���ϒl: {average_entropy}")




F_final = np.array([mode(arr)[0] for arr in zip(*all_F)])
#print(F_final)

# �Ƃ���ŁA���̌v�Z���ʂł���F�����S�Ȃ���m���x���s��Y1�Ƃǂꂾ��������������؂��Ă݂���
mask = Y1 != F_final


# �}�X�N����True�̐����J�E���g�i�܂�A�v�f���قȂ���j
num_diff = np.count_nonzero(mask)


# ����ɂ��A��ĕ��@���A���Ƃ̃��x�����ǂ��܂ōČ��i���x���`�d�����̗ǂ��O���t�j�o���Ă����������؂���
# ���t��7���Ȃ�7.4�p�[�Z���g(�`6.2�p�[�Z���g)�̌덷�I�I ���t��3���Ȃ�11.5�p�[�Z���g
# �v�f�̐��� 950 x 5 = 4750
print("���������m���x���s��Ƃ̍������ǂꂾ�����邩:", num_diff)



# �Ō�͏W�v�B���ꂽ�m�C�Y�̂����A�ǂꂾ���̃��x���𓖂Ă�ꂽ���i�ύX�����������j

def count_matching_rows(matrix1, matrix2, row_indices):
    """
    �w�肵���s�ԍ��̍s�̗v�f�̕��т����S�Ɉ�v���Ă��鐔���J�E���g����
    Args:
        matrix1, matrix2: ��r����2�̍s��
        row_indices: ��r����s�̃C���f�b�N�X�̃��X�g
    Returns:
        ���S�Ɉ�v�����s�̐�
    """

    # �w�肳�ꂽ�s�݂̂����o��
    selected_rows1 = matrix1[row_indices]
    selected_rows2 = matrix2[row_indices]

    # �e�s�̔�r���ʂ�True/False�ŕ\��
    comparison_result = (selected_rows1 == selected_rows2).all(axis=1)

    # True�̐����J�E���g
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
    �v�f���قȂ�s�̃C���f�b�N�X��Ԃ��֐�

    Args:
        Y0: numpy.ndarray
        F: numpy.ndarray

    Returns:
        list: �v�f���قȂ�s�̃C���f�b�N�X�̃��X�g
    """

    different_indices = []
    for i in range(Y0.shape[0]):
        if not np.array_equal(Y0[i], F[i]):
            different_indices.append(i)
    return different_indices


different_indices = find_different_rows(Y1, F)
#print("�v�f���قȂ�s�̃C���f�b�N�X:", different_indices)

set1 = set(changed_row)
set2 = set(different_indices)

#���ʗv�f�A���Ȃ킿�A��X�̎����Ō��o�Ɏ��s�����m�[�h�̃��X�g�i�s�ԍ��j
common_elements = set1.intersection(set2)
print("���ʗv�f:", list(common_elements))

#changed_row�i�̏W���^�jset1���狤�ʗv�f�i�̏W���^�jcommon_elements�������������W�����A�����Ō��o�ɐ��������m�[�h�ƂȂ�B
set_success = set1 - common_elements
print("����", list(set_success))



variances = []
for row_idx in changed_row:
    row_data = F0[row_idx, :]  # �w��s�̃f�[�^���擾
    variance = np.var(row_data)
    variances.append(variance)



print("���o��������label score�̕��U", variances)

mean_variance = np.mean(variances)

print("���o��������label score�̕��U�̕��ϒl", mean_variance)


unchanged_row = list(common_elements)

variances2 = []
for row_idx in unchanged_row:
    row_data = F0[row_idx, :]  # �w��s�̃f�[�^���擾
    variance = np.var(row_data)
    variances2.append(variance)


print("���o���s����label score�̕��U", variances2)

mean_variance2 = np.mean(variances2)

print("���o���s����label score�̕��U�̕��ϒl", mean_variance2)


# ��������͍l�@�̂��߁A���o�Ɏ��s�����m�[�h��PageRank�ƁA���o�ł����m�[�h��PageRank���r����
pr = nx.pagerank(G,alpha=0.75,weight='weight')

# �m�[�hID�ƍs�ԍ��̑Ή��������Ɋi�[
node_to_index = {node: index for index, node in enumerate(G.nodes)}

# ���o�Ɏ��s����list���̍s�ԍ��ɑΉ�����PageRank�X�R�A�𒊏o
result1 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in common_elements}

# �X�R�A�̃��X�g���쐬
scores = [score for node, (score, _) in result1.items()]

# �X�R�A�̕��ϒl���v�Z
average_score = sum(scores) / len(scores)

print("���o���s��PR�X�R�A�̕��ϒl:", round(average_score,6))
#print(result1)

# �t�ɁA���o�ɐ�������list���̍s�ԍ��ɑΉ�����PageRank�X�R�A�𒊏o
result2 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in set_success}

# �X�R�A�̃��X�g���쐬
scores = [score for node, (score, _) in result2.items()]

# �X�R�A�̕��ϒl���v�Z
average_score = sum(scores) / len(scores)

print("���o������PR�X�R�A�̕��ϒl:", round(average_score,6))
#print(result2)


'''
# ���x���s��F�ɏ]���ăO���tG�̃m�[�h���x�����X�V
for i, node in enumerate(G.nodes()):
    # F�̊e�s��one-hot�x�N�g���Ȃ̂ŁA�ő�l�̃C���f�b�N�X�����x���ɑΉ�����
    predicted_label_index = np.argmax(F[i])
    # ���x�����X�glabels����Ή����郉�x�����擾
    predicted_label = ['SD', 'NC', 'NS', 'CS', 'MD', 'LD', 'NI'][predicted_label_index]
    G.nodes[node]['label'] = predicted_label  # �m�[�h���x�����X�V

# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)

# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)

entropy_values_after = [] # �G���g���s�[�l���i�[���郊�X�g

# �X�V���ꂽ���x�����z���N���[�N���Ƃɏo��
for clique in sorted_cliques:
    # ���x�����J�E���g
    label_counts = Counter(G.nodes[node]['label'] for node in clique)
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # ���x�����z��\��
    print(len(clique), end=" ")  # �m�[�h���\��
    print("{", end="")
    for i, node in enumerate(clique):
        label = G.nodes[node]['label']  # �X�V���ꂽ�m�[�h���x�����擾
        if i < len(clique) - 1:
            print(", ", end="")
    print("}", end="  ")

    print("���x����:", end=" ")
    for label, count in sorted_label_counts:
        print(f"{label}:{count}", end=" ")


    # �G���g���s�[���v�Z���ĕ\��
    entropy = calculate_entropy(clique, {node: G.nodes[node]['label'] for node in clique})
    print(f"�G���g���s�[: {entropy}")  # �G���g���s�[��\��
    entropy_values_after.append(entropy) # ���X�g�ɒǉ�

# ���ׂẴN���[�N�̃G���g���s�[�̕��ϒl���v�Z
average_entropy = np.mean(entropy_values_after)
print(f"�S�N���[�N�̃G���g���s�[���ϒl: {average_entropy}")
'''
