# 6��8���BDSAA����Label commonization �����J�n - Herlev

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

# Herlev��JSON�t�@�C��(917��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
with open('./HFA_merged.json', 'r') as f:
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

def create_lle_graph(node_vectors, n_components, n_neighbors, metric='cosine'):

    # �m�[�h�x�N�g����NumPy�z��ɕϊ�
    X = np.array(node_vectors)

    # LLE�ɂ�鎟���팸
    embedding = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    X_transformed = embedding.fit_transform(X)

    # �᎟����Ԃɂ�����m�[�h�Ԃ̋����Ɋ�Â��ăG�b�W�𒣂�
    # ������threshold�ȉ��̃m�[�h���m�ɃG�b�W�𒣂�悤�ɂ���
    threshold = 0.4 # ���̒l�͏���������ƌv�Z�Ɏ��s����̂Œ���
    G_lle = nx.Graph()
    for i in range(len(X_transformed)):
        for j in range(i+1, len(X_transformed)):
            distance = np.linalg.norm(X_transformed[i] - X_transformed[j])
            if distance <= threshold:
                G_lle.add_edge(i, j)

    return G_lle

'''
# LLE�ō�����O���t
G_lle = create_lle_graph(list(data.values()), 100, 15, 'cosine')
'''

# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â���)
for i in range(len(similarity_matrix)):
    for j in range(i+1):
        # ���G�b�W�ގ��x��臒l�𒲐��@0.75 �Ȃ� �G�b�W��: 4362
        if similarity_matrix[i, j] > 0.75:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])

num_edges = G.number_of_edges()
print("�G�b�W��:", num_edges)

degrees = dict(G.degree())
#degrees = dict(G_lle.degree())

# �����̍��v���v�Z
sum_of_degrees = sum(degrees.values())


# �m�[�h�����擾
num_nodes = G.number_of_nodes()

'''
# ���ώ������v�Z
average_degree = sum_of_degrees / num_nodes
print("���ώ���:", average_degree)

isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
'''

# �e�m�[�h�̎��������X�g�Ɋi�[
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

# x��: �x�� (log�X�P�[��)
#x = np.arange(min(degree_sequence), max(degree_sequence) + 1)

# y��: �x�������m�[�h�� (log�X�P�[��)
#y = [list(degree_sequence).count(i) for i in x]


# �O���t�I�u�W�F�N�gG����אڍs����uS0�v�Ƃ��čs��ɕϊ�
# ���̂����ŁA���f���w�K�����邽�߁A���l�f�[�^��0����1�̊ԂŔ񕉂̎����ɕϊ����Čv�Z�\�ɂ��Ă���
#S0 = nx.adjacency_matrix(G)
#S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array
# print(S)
# G_lle

S0 = nx.adjacency_matrix(G)
S = minmax_scale(S0.toarray()) # Use minmax_scale from sklearn.preprocessing and convert S0 to a dense array

# ��������͊��m���x���Ŗ��߂�Y1�i917 x 7�j�����B
# �ŏ����牽�s�ڂ��܂ł͓����p�^���̍s�ɂȂ邪�A���̂��߂̊e�p�^�[���̍s���ƑΉ�����l�̃��X�g
# �ォ�珇��LD(182),MD(146),CS(150),SD(197),NC(98),NS(74),NI(70)�̌v917
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

# ��̃��X�g���쐬���Ă���
matrix_list = []

# �e�p�^�[���ŌJ��Ԃ��A�s�𐶐��������X�g�����
for num_rows, row_values in patterns:
    # �w�肳�ꂽ�s�����̍s�𐶐����A���X�g�ɒǉ�
    matrix_list.extend([row_values] * num_rows)

# ��������X�g���ANumPy�z��ɕϊ�
# ���m���x���s��(����l)�uY1�v�����i���ׂẴm�[�h�Ƀ��x�����t�^����Ă���j
Y1 = np.array(matrix_list)


def noise_input(matrix, max_row=675):
    """
    �s���1�s�ڂ���max_row�s�ڂ܂ł͈̔͂ŁA10%�̍s�������_���ɑI�����A����̗v�f��u������֐�
    Args:
        matrix: �����Ώۂ̍s��iY1�j
        max_row: �m�C�Y�𓱓�����ő�s�� (�f�t�H���g: 675)
    Returns:
        ������̍s��, �����_���ɑI�����ꂽ�s�̃C���f�b�N�X�̃��X�g
    """

    # �w�肳�ꂽ�s���܂ł͈̔͂ŏ���
    num_rows = min(max_row, matrix.shape[0])

    # �m�C�Y�����������@�����_���ɑI������s�̃C���f�b�N�X��SG�l�Őݒ�B0.1�Ȃ�10%����
    random_indices = np.random.choice(num_rows, int(num_rows * 0.3), replace=False)

    # �I�����ꂽ�s�ɑ΂��ď���
    for i in random_indices:
        # 1��0�ɒu��
        matrix[i, matrix[i] == 1] = 0

        # �c��̗v�f���烉���_����1��I������1�ɂ���
        zero_indices = np.where(matrix[i] == 0)[0]
        random_zero_idx = np.random.choice(zero_indices)
        matrix[i, random_zero_idx] = 1

    return matrix, random_indices

# ��������͎����̏����Ƃ��āA�����ł��銮�S�Ȃ���m���x���s��Y1����A�킴��
# ���������x�����ԈႦ�����ʁi���x���m�C�Y�����������̂𐶐��j��
# �s����uY0�v�Ƃ��A�ύX���ꂽ�s��changed_row�Ƃ��ďo�͂��Ă���
Y0, changed_row = noise_input(Y1)

changed_row.sort()
print(changed_row)
print("�m�C�Y�v�f��",len(changed_row))

##### 7���ނ�6���ނɏk�ނ�����֐�####
def reduce_matrix(Y):
    """
    �s��Y��6��ڂ�7��ڂ��k�񂷂�֐�
    Args:
        Y: 917�s7���numpy.ndarray
    Returns:
        Z: 917�s6���numpy.ndarray
    """

    # 6��ڂ�7��ڂ𔲂��o���A�ǂ��炩�����1�ł����1�A�����łȂ����0�ɂ���
    reduced_col = np.any(Y[:, 5:7], axis=1).astype(int)

    # ���̍s��Y����6��ڂ��폜���A�V�������ǉ�
    Z = np.delete(Y, 6, axis=1)
    Z = np.column_stack((Z, reduced_col.reshape(-1, 1)))

    return Z

# ���x���m�C�Y�̓�����Y0�ɑ΂��āA�k�ފ֐����Ăяo����Z���擾�@6��ɏk�ނ��Ă���
#Z = reduce_matrix(Y0)
#print(Z)


##### 7���ނ�5���ނɏk�ނ�����֐�####
def reduce_matrix_modified(Y):
    """
    �s��Y�̍��[����5��ځA6��ځA7��ڂ��k�񂷂�֐�
    Args:
        Y: 917�s7���numpy.ndarray
    Returns:
        Z_5: 917�s5���numpy.ndarray
    """

    # 5��ځA6��ځA7��ڂ𔲂��o���A�ǂꂩ��ł�1�������1�A�����łȂ����0�ɂ���
    reduced_col = np.any(Y[:, 4:7], axis=1).astype(int)

    # ���̍s��Y����5��ځA6��ځA7��ڂ��폜���A�V�������ǉ�
    Z_5 = np.delete(Y, [4, 5, 6], axis=1)
    Z_5 = np.column_stack((Z_5, reduced_col.reshape(-1, 1)))

    return Z_5

Z_5 = reduce_matrix_modified(Y0)
#print(Z_5)

# ��������1. ���s���B
num_trials = 11

#### ��������Y0�i7��̊��S�Łj�̃��x���`�d�@���������t�����̂��߃A���T���v��

# �e���s�̌��ʂ��i�[���郊�X�g�i11�̍s��F���i�[���郊�X�g�j
all_F = []

#### �������烉�x���`�d�i5��ɏk�ނ���Z_5�ɑ΂�����́j�@���������t�����̂��߃A���T���v��
all_F_red = []

# ������������J�n�B�����m�[�h��I�肷�邽�߁A�����_���Ƀ[���ɂ���
# �������x���s��Y2�����i�����11�񎎍s����B�Ȃ�111��܂Ŏ����Ă��D�ʂȐ��x����͌����Ȃ������j

for _ in range(num_trials):

 # �s�ԍ����Ƃ�臒l�ݒ�
 thresholds[:675] = 0.2  # �ُ�J�e�S���S����:675�B1�s�ڂ���675�s�ڂ܂ł�0.2�ɕύX�@�����80%�̏������x�����g���Ƃ������Ɓ�����
 thresholds[675:917] = 0.8  # ����J�e�S��1���ށF242�B676�s�ڂ���917�s�ڂ܂ł�0.4�ɕύX.�����60%�̏������x�����g���Ƃ������Ɓ@������

 # �s��Y�̍쐬
 Y3 = np.array([row if np.random.rand() > thresholds[i] else np.zeros_like(row) for i, row in enumerate(Z_5)])

 # �������烉�x���`�d�̎����v�Z�B
 # ��SG�l�BSet alpha
 alpha = 0.014

 # Calculate F0
 I = np.eye(917)
 inv_term = np.linalg.inv(I - alpha * S)
 F1 = inv_term.dot(Y3)

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
 F_red = process_matrix(F1) # 5��ɏk�ނ�������
 #print(F)
 all_F_red.append(F_red)

F_final_red = np.array([mode(arr)[0] for arr in zip(*all_F_red)])
#print(F_final_red)
#### �����܂ł̉򂪃��x���`�d

Y5 = Y0[:, :5] # Y0�̓m�C�Y����������7��̊��S�ŁB�����������5�񂾂��؂�o�����̂�Y5
#F5 = F_final[:, :5]  # 7��̊��S�ł̌v�Z���ʂ�F_final�B�����������5�񂾂��؂�o�����̂�F5
#F5_red = F_final_red[:, :5] # 6��ɏk�ނ����Ƃ��̌v�Z���ʂ�F_final_red�B�����������5�񂾂��؂�o�����̂�F5_red

#print("Y5", Y5.shape[0], Y5.shape[1])
#print("F_final_red", F_final_red.shape[0], F_final_red.shape[1])


def compare_rows(F, Y):
  return np.where(~(F == Y).all(axis=1))[0]

diff_rows = compare_rows(F_final_red, Y5)
print("�s��v�̍s�ԍ�:", diff_rows)

def extract_numbers_less_than_675_np(numbers):
  """
  NumPy�z�񂩂�675�ȉ��̐��l�݂̂𒊏o����֐�
  Args:
    numbers: NumPy�z��
  Returns:
    675�ȉ��̐��l��NumPy�z��
  """
  return numbers[numbers <= 675]

ext_num = extract_numbers_less_than_675_np(diff_rows)
print(ext_num)
print("���o�����v�f��",len(ext_num))

common_elements = set(changed_row) & set(diff_rows)
list_common = list(common_elements)
list_common.sort()
print(list_common)
print("�ǂ���ɂ��܂܂��v�f��",len(common_elements))


variances = []
for row_idx in common_elements:
    row_data = F1[row_idx, :]  # �w��s�̃f�[�^���擾
    variance = np.var(row_data)
    variances.append(variance)

mean_variance = np.mean(variances)
print("���o��������label score�̕��U�̕��ϒl", mean_variance)

fail_rows = list(set(changed_row) - set(common_elements))

for row_idx in fail_rows:
    row_data = F1[row_idx, :]  # �w��s�̃f�[�^���擾
    variance = np.var(row_data)
    variances.append(variance)

mean_variance = np.mean(variances)
print("���o���s����label score�̕��U�̕��ϒl", mean_variance)

def calculate_precision_recall(diff_rows, changed_rows):
    """
    �K�����ƍČ������v�Z����֐�
    Args:
        diff_rows (list): ���f�������Ɨ\�������v�f�̃��X�g
        changed_rows (list): ���ۂɌ���Ă����v�f�̃��X�g
    Returns:
        tuple: �K�����ƍČ����̃^�v��
    """
    # �^�z�� (True Positive): ���f�������Ɨ\�����A���ۂɌ���Ă����v�f
    true_positives = set(diff_rows).intersection(set(changed_rows))
    # �U�z�� (False Positive): ���f�������Ɨ\���������A���ۂɂ͐������v�f
    false_positives = set(diff_rows) - true_positives
    # �U�A�� (False Negative): ���f�����������\���������A���ۂɂ͌���Ă����v�f
    false_negatives = set(changed_rows) - true_positives
    # �K���� (Precision) = �^�z�� / (�^�z�� + �U�z��)
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    # �Č��� (Recall) = �^�z�� / (�^�z�� + �U�A��)
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0

    return precision, recall

precision, recall = calculate_precision_recall(ext_num, changed_row)
print("Precision:", precision)
print("�Č���:", recall)

# ��������ǉ�����R�[�h

print("\n--- ���x���C�����x�̕]�� ---")

# 1. �^�̐������x���i�m�C�Y�����O��Y1�j��5��ɏk�ނ�����
# Y1�̓m�C�Y�����O�̊��S�Ȑ������x���i7��j
Y1_red_for_eval = reduce_matrix_modified(Y1)

# 2. ���x���`�d��̍ŏI�\������ F_final_red �i5��j�Ɣ�r

# �S�̂̃��x���C�����x
correctly_corrected_all = np.sum((F_final_red == Y1_red_for_eval).all(axis=1))
total_rows = Y1_red_for_eval.shape[0]
overall_correction_accuracy = correctly_corrected_all / total_rows
print(f"�S�̂̃��x���C�����x: {overall_correction_accuracy:.4f}")

# �m�C�Y���������ꂽ�s�ichanged_row�j�ɂ����郉�x���C�����x
# F_final_red��Y1_red_for_eval��changed_row�̕������r
corrected_noisy_rows = 0
for idx in changed_row:
    # ���x���`�d�̌��� (F_final_red[idx]) ���A���̐��������x�� (Y1_red_for_eval[idx]) �ƈ�v���邩
    if np.array_equal(F_final_red[idx], Y1_red_for_eval[idx]):
        corrected_noisy_rows += 1

total_noisy_rows = len(changed_row)
noise_correction_accuracy = corrected_noisy_rows / total_noisy_rows if total_noisy_rows > 0 else 0

print(f"�m�C�Y���������ꂽ�s�ɂ����郉�x���C�����x: {noise_correction_accuracy:.4f}")

# ����J�e�S���i675�s�ڈȍ~�j�ɂ����郉�x���C�����x�i�m�C�Y����������Ă��Ȃ������j
# Z_5�̍쐬���ɐ���J�e�S����臒l��0.4�Ȃ̂ŁA���̕����ɂ��������x�����[���ɂȂ����ӏ�������
normal_category_start_idx = 675
corrected_normal_rows = 0
total_normal_rows = 0
for idx in range(normal_category_start_idx, Y1_red_for_eval.shape[0]):
    total_normal_rows += 1
    if np.array_equal(F_final_red[idx], Y1_red_for_eval[idx]):
        corrected_normal_rows += 1

normal_correction_accuracy = corrected_normal_rows / total_normal_rows if total_normal_rows > 0 else 0
print(f"����J�e�S���i�m�C�Y�Ȃ��j�ɂ����郉�x���C�����x: {normal_correction_accuracy:.4f}")