# Herlev�̃f�[�^�Bprompt: 643�s��Key��Value����\�������merged_data.json��ǂݍ��݂܂��BKey���O���t�̃m�[�h�Ƃ��AValue�͂��̃m�[�h�̓����x�N�g���i768�����j��\���܂��B����merged_data.json�̍ŏ���128�s�͎���Value�ǂ������W�܂��Ă��܂��B���Ȃ킿�Alight_dysplastic�Ƃ��Č݂��ɗގ������f�[�^�ł��B����103�s�́Amoderate_dysplastic�Ƃ��Č݂��ɗގ������f�[�^�ł��B����69�s��normal_columnar�Ƃ��Č݂��ɗގ������f�[�^�ł��B����105�s��carcinoma_in_situ�Ƃ��Č݂��ɗގ������f�[�^�ł��B����51�s��normal_superficie�Ƃ��Č݂��ɗގ������f�[�^�ł��B����49�s��normal_intermediate�Ƃ��Č݂��ɗގ������f�[�^�ł��B�Ō��138�s��severe_dysplastic�Ƃ��Č݂��ɗގ������f�[�^�ł��B���ꂼ��̍s�́A���ꂼ�ꂱ��7�ɕ��ނ��ꂽ���ŗގ����������Ȃ�悤�ɁAKey�ǂ������m�[�h�����O���t����肽���B����ɓK����悤��Key�ǂ����̋������v�Z����R�[�h�������āB

import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale # Import minmax_scale from sklearn.preprocessing
from scipy import stats
from scipy.stats import mode

# Herlev��JSON�t�@�C��(634��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
with open('./merged_data.json', 'r') as f:
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

# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.75 �Ȃ� �G�b�W��: 4362
        if similarity_matrix[i, j] > 0.74:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("�G�b�W��:", num_edges)

'''
degrees = dict(G.degree())

# �����̍��v���v�Z
sum_of_degrees = sum(degrees.values())

# �m�[�h�����擾
num_nodes = G.number_of_nodes()

# ���ώ������v�Z
average_degree = sum_of_degrees / num_nodes
print("���ώ���:", average_degree)


# �e�m�[�h�̎��������X�g�Ɋi�[
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# �������z���q�X�g�O�����ŉ���
plt.hist(degree_sequence, bins=20, color='blue')
plt.title("degree distribution")
plt.ylabel("# of node")
plt.xlabel("degree")
plt.show()
'''


'''
# �f�[�^�̕��ދ��E
boundaries = [0, 128, 231, 300, 405, 456, 505, 643]
labels = ["light_dysplastic", "moderate_dysplastic", "normal_columnar", "carcinoma_in_situ", "normal_superficie", "normal_intermediate", "severe_dysplastic"]

# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â��āA�������J�e�S�����݂̂ł������񎎂��Ă݂����́B�{�����ł͎g��Ȃ�)
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):  # i+1����n�߂邱�Ƃŏd�������
        # �J�e�S���̔���
        category_i = -1
        category_j = -1
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= i < boundaries[k + 1]:
                category_i = k
            if boundaries[k] <= j < boundaries[k + 1]:
                category_j = k

        # �����J�e�S�������ގ��x��臒l�ȏ�̏ꍇ�ɃG�b�W��ǉ�
        if category_i == category_j and similarity_matrix[i, j] > 0.75:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
'''


# �O���t�I�u�W�F�N�gG����אڍs����uS0�v�Ƃ��čs��ɕϊ�
# ���̂����ŁA���f���w�K�����邽�߁A���l�f�[�^��0����1�̊ԂŔ񕉂̎����ɕϊ����Čv�Z�\�ɂ��Ă���
S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)


# ��������͊��m���x���Ŗ��߂�Y1�i634 x 7�j�����B
# �ŏ����牽�s�ڂ��܂ł͓����p�^���̍s�ɂȂ邪�A���̂��߂̊e�p�^�[���̍s���ƑΉ�����l�̃��X�g
# �ォ�珇��LD(128),MD(103),NC(69),CS(105),NS(51),NI(49),SD(138)�̌v634
patterns = [
    (128, [1,0,0,0,0,0,0]),
    (103, [0,1,0,0,0,0,0]),
    (69, [0,0,1,0,0,0,0]),
    (105, [0,0,0,1,0,0,0]),
    (51, [0,0,0,0,1,0,0]),
    (49, [0,0,0,0,0,1,0]),
    (138, [0,0,0,0,0,0,1])
]

# ��̃��X�g���쐬���Ă���
matrix_list = []

# �e�p�^�[���ŌJ��Ԃ��A�s�𐶐��������X�g�����
for num_rows, row_values in patterns:
    # �w�肳�ꂽ�s�����̍s�𐶐����A���X�g�ɒǉ�
    matrix_list.extend([row_values] * num_rows)

# ��������X�g���ANumPy�z��ɕϊ�
# ���m���x���s��(����l)�uY1�v�����i���ׂẴm�[�h�Ƀ��x�����t�^����Ă���j
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

    # �s�����擾�i634�s�j
    num_rows = matrix.shape[0]

    # ��SG�l�B�����_���m�C�Y�̊����������_���ɑI���@0.1�Ȃ�S�̂�10%�Ɍ��������
    random_indices = np.random.choice(num_rows, int(num_rows * 0.15), replace=False)

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

'''
def show_matching_rows(Y0, F, changed_row):
    """
    �v�f�������s�̐����J�E���g���A��v�����s�̃C���f�b�N�X��Ԃ��֐�

    Args:
        Y0: numpy.ndarray
        F: numpy.ndarray
        changed_row: int (���g�p)

    Returns:
        list: ��v�����s�̃C���f�b�N�X�̃��X�g
    """

    matching_indices = []
    for i in range(Y0.shape[0]):
        if np.array_equal(Y0[i], F[i]):
            matching_indices.append(i)
    return matching_indices
'''


# ��������͎����̏����Ƃ��āA�����ł��銮�S�Ȃ���m���x���s��Y1����A�킴��
# ���������x�����ԈႦ�����ʁi���x���m�C�Y�����������̂𐶐��j��
# �s����uY0�v�Ƃ��A�ύX���ꂽ�s��changed_row�Ƃ��ďo�͂��Ă���
Y0, changed_row = modify_matrix(Y1)

print(changed_row)

# ��������1. ���s���B
num_trials = 11

# �e���s�̌��ʂ��i�[���郊�X�g�i11�̍s��F���i�[���郊�X�g�j
all_F = []

# ������������J�n�B�����m�[�h��I�肷�邽�߁A�����_���Ƀ[���ɂ���
# �������x���s��Y2�����i�����11�񎎍s����B�Ȃ�111��܂Ŏ����Ă��D�ʂȐ��x����͌����Ȃ������j

for _ in range(num_trials):
 # ��SG�l�B�Ⴆ�΁@> 0.3 �Ƃ������Ƃ͑S�̂�3�����[���Ƃ��āA7�������̂܂܏����f�[�^�Ƃ��Ďc���Ƃ�������
 # Y2�͎����i���x���`�d�v�Z�j�̂��ߕ֋X�I�Ɉꎞ�쐬��������
 Y2 = np.array([row if np.random.rand() > 0.5 else np.zeros_like(row) for row in Y0])
 #print(Y2)

 # �������烉�x���`�d�̎����v�Z�B
 # ��SG�l�BSet alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(643)
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

F_final = np.array([mode(arr)[0] for arr in zip(*all_F)])
print(F_final)

# �Ƃ���ŁA���̌v�Z���ʂł���F�����S�Ȃ���m���x���s��Y1�Ƃǂꂾ��������������؂��Ă݂���
mask = Y1 != F_final


# �}�X�N����True�̐����J�E���g�i�܂�A�v�f���قȂ���j
num_diff = np.count_nonzero(mask)


# ����ɂ��A��ĕ��@���A���Ƃ̃��x�����ǂ��܂ōČ��i���x���`�d�����̗ǂ��O���t�j�o���Ă����������؂���
# ���t��7���Ȃ�7.4�p�[�Z���g(�`6.2�p�[�Z���g)�̌덷�I�I ���t��3���Ȃ�11.5�p�[�Z���g
# �v�f�̐��� 634 x 7 = 4438
print("���������m���x���s��Ƃ̍������ǂꂾ�����邩:", num_diff)



# �Ō�͏W�v�B���ꂽ�m�C�Y�i�Ӑ}�I�ɂ킴�ƌ����changed_row�j�̂����A
# �ǂꂾ���̃��x���𓖂Ă�ꂽ���i�ύX�����������j

def count_matching_rows(matrix1, matrix2, row_indices):
    """
    �w�肵���s�ԍ��̍s�̗v�f�̕��т����S�Ɉ�v���Ă��鐔���J�E���g����
    Args:
        matrix1, matrix2: ��r����2�̍s��
        row_indices: ��r����s�̃C���f�b�N�X�̃��X�g(���Ȃ킿changed_row�͈̔͂Ŕ�r)
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

# �����Ă��ꂪ�A���̎����ł����Ƃ��s�������������ʁBChannged_row�͈̔͂łӂ���
# �s��̍s����v�����A�Ƃ������Ƃ́A�v�Z����F���AY0�ō������ꂽ���x���m�C�Y�����o�ł����Ƃ�������
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

print("���o���s��PR�X�R�A�̕��ϒl:", average_score)
#print(result1)

# �t�ɁA���o�ɐ�������list���̍s�ԍ��ɑΉ�����PageRank�X�R�A�𒊏o
result2 = {node: (score, node_to_index[node]) for node, score in pr.items() if node_to_index[node] in set_success}

# �X�R�A�̃��X�g���쐬
scores = [score for node, (score, _) in result2.items()]

# �X�R�A�̕��ϒl���v�Z
average_score = sum(scores) / len(scores)

print("���o������PR�X�R�A�̕��ϒl:", average_score)
#print(result2)

'''
# �X�v�����O���C�A�E�g
pos = nx.spring_layout(G, k=0.85, iterations=130, weight='weight')
# pos = nx.circular_layout(G)

# �m�[�h�ƃG�b�W�̕`��
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=0.1)

# ���x���\���𒲐�
# �m�[�h�����������߁A���x���\���͏ȗ��܂��͈ꕔ�̂ݕ\������
for node, (x, y) in pos.items():
     if node in nodes[:10]:  # ��F�擪10�̃m�[�h�̂݃��x����\��
         plt.text(x, y, node, fontsize=4, ha='center', va='center')

plt.axis('off')
plt.show()
'''