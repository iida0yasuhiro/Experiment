# ���ׂẴJ�e�S���ɓ��������Ńm�C�Y����ꂽ�ꍇ�@5��7������
'''
# Package installation (hidden on docs website).
dependencies = ["sklearn", "matplotlib", "pandas", "networkx", "scipy"] # cleanlab���폜

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
#5��7����SIPaKMeD�ŃO���t�N���[�N�����BICAHS�������x���`�d
#5��7���B�������x���������_���I������ۂɁA���𒲐��ł���悤�ɂ����B
import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity # �R�T�C���ގ��x�v�Z�̂��߂Ɏg�p
from collections import Counter # ���x�����J�E���g�̂��߂Ɏg�p
from sklearn.preprocessing import normalize # ���x���`�d�@�̂��߂Ɏg�p
import networkx as nx # �O���t�����̂��߂Ɏg�p
import math # �G���g���s�[�v�Z�̂��߂Ɏg�p
from scipy.stats import mode # ���x���`�d�@�̂��߂Ɏg�p
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score # �]���w�W�v�Z�̂��߂Ɏg�p

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

# JSON�t�@�C��(950��node��Key�i�t�@�C�����j-Value�i�����x�N�g���j�Ŋi�[)��ǂݍ���
# �t�@�C���p�X�͊��ɍ��킹�ēK�X�ύX���Ă�������
try:
    with open('./S_merged_data.json', 'r') as f:
        data = json.load(f)
    nodes = list(data.keys())
    feature_vectors = np.array(list(data.values()))
    print(f"�f�[�^�t�@�C�� './S_merged_data.json' ��ǂݍ��݂܂����B�m�[�h��: {len(nodes)}")

except FileNotFoundError:
    print("�G���[: �f�[�^�t�@�C�� './S_merged_data.json' ��������܂���B�t�@�C���p�X���m�F���Ă��������B")
    # �ȍ~�̏����͎��s�ł��Ȃ����߁A�����ŏI��
    exit()
except Exception as e:
    print(f"�G���[: �f�[�^�t�@�C���ǂݍ��ݒ��ɃG���[���������܂��� - {e}")
    # �ȍ~�̏����͎��s�ł��Ȃ����߁A�����ŏI��
    exit()


# �R�T�C���ގ��x���v�Z
similarity_matrix = cosine_similarity(feature_vectors)
print("�R�T�C���ގ��x�s����v�Z���܂����B")

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
        # labels��������m�[�h�̃��x�����擾
        label = labels.get(node)
        if label is not None:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    entropy = 0
    num_nodes = len(clique)
    if num_nodes == 0: # �N���[�N����̏ꍇ�̓G���g���s�[0
        return 0

    for count in label_counts.values():
        probability = count / num_nodes
        if probability > 0: # log2(0)�������
            entropy -= probability * math.log2(probability)

    return entropy

def find_maximal_cliques(graph):
    """
    �O���t���̋ɑ�N���[�N�����ׂČ�����֐� (�m�[�h��3�ȏ�A���̋ɑ�N���[�N�ɕ�܂������̂͏���)
    �� networkx�̑g�ݍ��݊֐����g�p���A�m�[�h��3�ȏ�̃t�B���^�����O��K�p���܂��B

    Args:
        graph: networkx�O���t�I�u�W�F�N�g

    Returns:
        �m�[�h��3�ȏ�ŁA���̋ɑ�N���[�N�ɕ�܂���Ȃ��ɑ�N���[�N�̃��X�g
    """

    maximal_cliques = []
    # networkx�̑g�ݍ��݊֐����g�p���ċɑ�N���[�N�������I�Ɍ����܂�
    for clique in nx.find_cliques(graph):
        if len(clique) >= 3: # �m�[�h��3�ȏ�̃N���[�N��Ώ�
            maximal_cliques.append(set(clique)) # set�ɕϊ����ĕ�܊֌W�̔����e�Ղɂ���

    # ���̋ɑ�N���[�N�ɕ�܂����N���[�N�����O
    filtered_cliques = []
    for c1 in maximal_cliques:
        is_subset = False
        for c2 in maximal_cliques:
            if c1 != c2 and c1.issubset(c2):
                is_subset = True
                break
        if not is_subset:
            filtered_cliques.append(list(c1)) # ���X�g�ɖ߂�

    return filtered_cliques


# �O���t�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# �m�[�h�̒ǉ� (�摜�t�@�C����)
G.add_nodes_from(nodes)

# �m�[�h���x���̃��X�g���쐬 (SIPaKMeD�̃N���X�Ɛ��Ɋ�Â��Ă��܂�)
# ����̓m�C�Y�����O�́u�^�̃��x���v�Ƃ��Ďg�p���܂��B
# �ォ�珇��ME(271),KO(232),DY(223),PA(108),SU(116)�̌v950

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



# �G�b�W�̒ǉ� (�ގ��x�Ɋ�Â���)
# �ގ��x�s��͂��łɌv�Z�ς� (similarity_matrix)
threshold = 0.74 # �ގ��x��臒l
edges_added = 0
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)): # �d��������邽�� j �� i+1 ����J�n
        if similarity_matrix[i, j] > threshold:
            G.add_edge(nodes[i], nodes[j], weight=similarity_matrix[i, j])
            edges_added += 1
print(f"�ގ��x {threshold} �ȏ�̃G�b�W�� {edges_added} �ǉ����܂����B")


# �ɑ�N���[�N��������
maximal_cliques = find_maximal_cliques(G)
print(f"�m�[�h��3�ȏ�̋ɑ�N���[�N�� {len(maximal_cliques)} �����܂����B")


# �ɑ�N���[�N���m�[�h�����������Ƀ\�[�g
sorted_cliques = sorted(maximal_cliques, key=len, reverse=True)
print(f"�ɑ�N���[�N���m�[�h���Ń\�[�g���܂����B")


# �e�N���[�N�̎x�z�I�ȃ��x�������m�[�h���������x���Ƃ��Ē��o
initial_labels = {} # ���x���`�d�̏������x���Ƃ��Ďg�p���鎫�� {�m�[�h��: ���x��}
clique_info = []
entropy_values = [] # �N���[�N�̃G���g���s�[�l���i�[

print("\n�e�N���[�N�̏���������...")
for i, clique in enumerate(sorted_cliques):
    # �N���[�N���̃m�[�h�̌��݂̃��x�����擾 (�m�C�Y������̃��x��)
    current_clique_labels = {node: G.nodes[node]['label'] for node in clique}

    # ���x���d�݂��v�Z (�N���[�N���̃G�b�W�̏d�݂̍��v)
    label_weights = {}
    for node in clique:
        label = current_clique_labels[node]
        if label not in label_weights:
            label_weights[label] = 0
        for neighbor in G.neighbors(node):
            if neighbor in clique: # �N���[�N���̗אڃm�[�h�݂̂��l��
                label_weights[label] += G[node][neighbor]['weight']

    # �x�z�I�ȃ��x�������� (���x���d�݂��ő�̃��x��)
    dominant_label = None
    if label_weights:
        dominant_label = max(label_weights, key=label_weights.get)
    else: # ���x���d�݂��v�Z�ł��Ȃ��ꍇ�i��F�G�b�W���Ȃ��N���[�N�j
        # �N���[�N���̃m�[�h�̃��x���̍ŕp�l���x�z�I�ȃ��x���Ƃ���
        clique_labels_list = list(current_clique_labels.values())
        if clique_labels_list:
             dominant_label_mode, _ = mode(clique_labels_list, keepdims=True) # keepdims=True�ŏ�ɔz���Ԃ��悤��
             dominant_label = dominant_label_mode[0]
        else: # �N���[�N����i����� find_maximal_cliques �ŏ��O�����͂��ł����O�̂��߁j
             dominant_label = None # �x�z�I�ȃ��x���Ȃ�


    # ���x�����J�E���g
    label_counts = Counter(current_clique_labels.values())
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

    # �G���g���s�[���v�Z���ă��X�g�ɒǉ�
    entropy = calculate_entropy(clique, current_clique_labels)
    entropy_values.append(entropy)

    # �x�z�I�ȃ��x�������m�[�h���������x�����Ƃ��đI��
    dominant_nodes_in_clique = [node for node in clique if current_clique_labels.get(node) == dominant_label]

    # �x�z�I�ȃ��x�������m�[�h�̒�����A�܂��������x���Ƃ��đI������Ă��Ȃ��m�[�h��D��I�ɑI��
    initial_node = None
    for node in dominant_nodes_in_clique:
        if node not in initial_labels:
            initial_node = node
            break

    # �x�z�I�ȃ��x�������m�[�h���S�ď������x���Ƃ��đI���ς݂̏ꍇ�A
    # �N���[�N���̍ŏ��̃m�[�h���������x���Ƃ��đI���i����͂��܂�]�܂����Ȃ���������܂���j
    if initial_node is None and dominant_nodes_in_clique:
         initial_node = dominant_nodes_in_clique[0]


    # �x�z�I�ȃ��x����������A�������m�[�h���I���ł����ꍇ�̂�initial_labels�ɒǉ�
    # ���� initial_labels �����x���`�d�́u�Œ肳�ꂽ�v�������x���Ƃ��ċ@�\���܂��B
    if dominant_label is not None and initial_node is not None:
         initial_labels[initial_node] = dominant_label


    # �ǉ�: �N���[�N�������X�g�ɒǉ�
    clique_info.append({
        "clique_id": i,
        "clique_size": len(clique),
        "dominant_label": dominant_label,
        "label_weights": label_weights,
        "label_counts": dict(sorted_label_counts), # Counter��dict�ɕϊ����ĕۑ�
        "entropy": entropy,
        "initial_node_for_ls": initial_node # ���x���`�d�Ɏg�������m�[�h (initial_labels�Ɋ܂܂��m�[�h)
    })

print(f"���x���`�d�̏������x���Ƃ��� {len(initial_labels)} �̃m�[�h��I�����܂����B")


def label_spreading(graph, initial_labels, alpha=0.1, max_iter=100, tol=1e-5):
    """���x���`�d�@��p���ă��x�����C������֐��i�������A���t�@�l�𒲐�����j"""

    # initial_labels���烆�j�[�N�ȃ��x���̃��X�g���쐬
    unique_labels = sorted(list(set(initial_labels.values())))
    if not unique_labels:
        print("�x��: initial_labels����ł��B���x���`�d�͎��s����܂���B")
        # �O���t�̑S�m�[�h�ɑ΂���None��Ԃ��������쐬
        return {node: None for node in graph.nodes()}


    label_indices = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    num_nodes = len(graph)
    nodes_list = list(graph.nodes()) # �m�[�h�̃��X�g���쐬���ăC���f�b�N�X�A�N�Z�X�\�ɂ���

    # �������x���s��̍쐬 (F)
    # F[i, j] �̓m�[�h i �����x�� j �����m���̏����l
    F = np.zeros((num_nodes, num_labels))
    # �������x�������m�[�h�ɑΉ�����s��1�ɐݒ�
    # initial_labels�Ɋ܂܂��m�[�h�́A���̃��x���̊m����1�A���̃��x���̊m����0�ƂȂ�
    # initial_labels�Ɋ܂܂�Ȃ��m�[�h�́A�����m���͑S��0�ƂȂ�
    for node, label in initial_labels.items():
        if node in nodes_list: # �O���t�ɑ��݂���m�[�h���m�F
            node_idx = nodes_list.index(node)
            if label in label_indices: # �L���ȃ��x�����m�F
                 label_idx = label_indices[label]
                 F[node_idx, label_idx] = 1

    # �O���t�̗אڍs��̍쐬 (W)
    # W[i, j] �̓m�[�h i �ƃm�[�h j �̊Ԃ̃G�b�W�̏d��
    W = nx.adjacency_matrix(graph, nodelist=nodes_list).toarray()

    # ���K�����ꂽ�אڍs�� (S) �̌v�Z
    # S = D^(-1/2) W D^(-1/2)
    # D �͑Ίp�s��ŁAD[i, i] �̓m�[�h i �̎����i�܂��͏d�݂̍��v�j
    row_sum = W.sum(axis=1)
    # �[�����Z��h��
    D_inv_sqrt = np.diag(np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0))
    S = np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)


    # ���x���`�d�̔����v�Z
    # F(t+1) = alpha * S * F(t) + (1 - alpha) * Y
    # Y �͏������x���s�� (�����l�̂�1�A���̑�0) - ���x�����Œ肳��镔��
    Y = np.copy(F) # �������x���s���Y�Ƃ��ĕێ�

    for iter_count in range(max_iter):
        F_new = alpha * np.dot(S, F) + (1 - alpha) * Y
        # ��������
        diff = np.linalg.norm(F_new - F)
        F = F_new
        if diff < tol:
            print(f"���x���`�d���������܂��� (iteration: {iter_count+1})")
            break
    else:
        print(f"�x��: ���x���`�d���ő唽���� ({max_iter}) �ɒB���܂������������܂���ł����B")


    # �C�����ꂽ���x���̎擾
    # �e�m�[�h�ɑ΂��āA�m�����ő�ƂȂ郉�x����I��
    # �������A�������x���Ƃ��ė^����ꂽ�m�[�h�̃��x���͌Œ肳��Ă���͂��Ȃ̂ŁA
    # ���̃m�[�h�̃��x���� initial_labels �̒l���g�p����̂���萳�m��������܂���B
    # �����ł͓`�d���F�s��̍ő�l���g�p���܂����Ainitial_labels��D�悷�郍�W�b�N���l�����܂��B
    new_labels_list = []
    for i in range(num_nodes):
        node = nodes_list[i]
        if node in initial_labels:
            # �������x���Ƃ��ė^����ꂽ�m�[�h�́A���̃��x�����ŏI���x���Ƃ���
            new_labels_list.append(initial_labels[node])
        else:
            # ����ȊO�̃m�[�h�́A�`�d��̊m�����ő�̃��x����I��
            if np.sum(F[i, :]) > 0: # �m���̍��v��0���傫���ꍇ�̂݃��x��������
                 new_labels_list.append(unique_labels[np.argmax(F[i, :])])
            else:
                 # �m�����S��0�̏ꍇ�ȂǁA���x��������ł��Ȃ��ꍇ��None�Ƃ���
                 new_labels_list.append(None)


    # �m�[�h���ƏC�����ꂽ���x����Ή������鎫�����쐬
    new_labels_dict = {nodes_list[i]: new_labels_list[i] for i in range(num_nodes)}

    return new_labels_dict


# ���x���`�d�@��p���ă��x�����C�� (�N���[�N�̎x�z�I�ȃ��x�����������x���Ƃ��Ďg�p)
propagated_labels_dict = label_spreading(G, initial_labels)

# �]���̌v�Z
# �]���́A�m�C�Y�������ꂽchanged_nodes�ɑ΂��čs���܂��B
# ���x���`�d�ɂ���āA�����̃m�[�h�̃��x�������̐��������x���ɂǂꂾ���߂�������]�����܂��B

# �]���Ɏg�p����^�̃��x���́A�m�C�Y�����O�� original_labels_dict �ł��B
# �]���Ɏg�p����\�����x���́A���x���`�d��� propagated_labels_dict �ł��B

# TP, TN, FP, FN ���v�Z
# �����ł̕]���́u�m�C�Y�������ꂽ�m�[�h�̃��x�������̃��x���ɖ߂������v����Ƃ��܂��B
# TP: �m�C�Y�������ꂽ�m�[�h�ŁA���x���`�d��Ɍ��̃��x���ɖ߂�������
# FN: �m�C�Y�������ꂽ�m�[�h�ŁA���x���`�d������̃��x���ɖ߂�Ȃ���������
# FP: �m�C�Y��������Ă��Ȃ��m�[�h�ŁA���x���`�d��Ɍ��̃��x������ς���Ă��܂�������
# TN: �m�C�Y��������Ă��Ȃ��m�[�h�ŁA���x���`�d������̃��x���̂܂܂���������

TP = 0 # True Positive: �m�C�Y��������A�`�d��Ɍ��̃��x���ɖ߂���
FN = 0 # False Negative: �m�C�Y��������A�`�d������̃��x���ɖ߂�Ȃ�����
FP = 0 # False Positive: �m�C�Y��������Ă��Ȃ����A�`�d��Ƀ��x�����ς����
TN = 0 # True Negative: �m�C�Y��������Ă��炸�A�`�d������x�����ς��Ȃ�����

# changed_nodes (�m�C�Y�������ꂽ�m�[�h) �ɂ��ĕ]��
for node in changed_nodes:
    actual_original_label = original_labels_dict[node] # �m�C�Y�����O�̐^�̃��x��
    predicted_propagated_label = propagated_labels_dict.get(node) # ���x���`�d��̃��x��

    if predicted_propagated_label is not None: # ���x���`�d�Ń��x�����t�^���ꂽ���m�F
        if predicted_propagated_label == actual_original_label:
            TP += 1 # �m�C�Y���C�����ꂽ�i���̃��x���ɖ߂����j
        else:
            FN += 1 # �m�C�Y���C������Ȃ������i���̃��x���ɖ߂�Ȃ������j
    # else: # ���x���`�d�Ń��x�����t�^����Ȃ������ꍇ�i�������ɂ����P�[�X�ł����j
    #     FN += 1 # ���̃��x���ɖ߂�Ȃ������Ƃ݂Ȃ�


# changed_nodes �ȊO�̃m�[�h (�m�C�Y��������Ă��Ȃ��m�[�h) �ɂ��ĕ]��
# �����̃m�[�h�̐^�̃��x���� original_labels_dict �Ɋi�[����Ă��܂��B
# ���x���`�d������̃��x���̂܂܂ł��邱�Ƃ��]�܂����ł��B
unchanged_nodes = [node for node in nodes if node not in changed_nodes]

for node in unchanged_nodes:
    actual_original_label = original_labels_dict[node] # �m�C�Y�����O�̐^�̃��x��
    predicted_propagated_label = propagated_labels_dict.get(node) # ���x���`�d��̃��x��

    if predicted_propagated_label is not None: # ���x���`�d�Ń��x�����t�^���ꂽ���m�F
        if predicted_propagated_label == actual_original_label:
            TN += 1 # ���x�������̂܂܈ێ����ꂽ
        else:
            FP += 1 # ���x�����ς���Ă��܂����i����ĕύX���ꂽ�j
    # else: # ���x���`�d�Ń��x�����t�^����Ȃ������ꍇ
    #     # �^�̃��x����������Ȃ��A�܂��͕]���ΏۊO�Ƃ���ȂǁA�󋵂ɉ����Ĕ��f���K�v
    #     pass # �����ł͕]���Ɋ܂߂Ȃ�


# ���ʂ�\��
print("\n���x���`�d�ɂ��m�C�Y�C���̕]�� (�N���[�N�̎x�z�I���x�����������x���Ƃ��Ďg�p):")
print("TP (�m�C�Y�C������):", TP)
print("FN (�m�C�Y�C�����s):", FN)
print("FP (����ă��x���ύX):", FP)
print("TN (���������x���ێ�):", TN)

total_evaluated_nodes = TP + FN + FP + TN
print('���v�]���m�[�h���F', total_evaluated_nodes)
print('�S�m�[�h���F', len(nodes))
# ���Ftotal_evaluated_nodes �́A���x���`�d��Ƀ��x�����t�^���ꂽ�m�[�h�̂����A
# changed_nodes �� unchanged_nodes �Ɋ܂܂��m�[�h�̍��v�ł��B
# ���x���`�d�Ń��x�����t�^����Ȃ������m�[�h�͕]���Ɋ܂܂�Ă��܂���B


# �K�����A�Č����A���x�AF�l���v�Z
# �����ł̓K�����ƍČ����́A�u�m�C�Y�������ꂽ�m�[�h���ǂꂾ�����̃��x���ɖ߂������v
# �Ƃ����ϓ_�ł̕]���w�W�ƂȂ�܂��B
# �K���� (Precision): �C�����ꂽ�Ɨ\�������m�[�h�iTP+FP�j�̂����A���ۂɃm�C�Y���C�����ꂽ�iTP�j����
# �Č��� (Recall): ���ۂɃm�C�Y�������ꂽ�m�[�h�iTP+FN�j�̂����A�m�C�Y���C�����ꂽ�iTP�j����
# ���x (Accuracy): �S�]���m�[�h�̂����A�������]�����ꂽ�m�[�h�iTP+TN�j�̊���

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# ���ʂ�\��
print("\n���x���`�d�ɂ��m�C�Y�C���̕]���w�W (�N���[�N�̎x�z�I���x�����������x���Ƃ��Ďg�p):")
print(f"�K���� (Precision): {precision:.4f}")
print(f"�Č��� (Recall): {recall:.4f}")
print(f"���x (Accuracy): {accuracy:.4f}")
print(f"F�l (F1-score): {f1:.4f}")


# �����_���ȏ������x�����g�p�����ꍇ�̕]��
# �V���Ƀ����_���ȏ������x���Z�b�g���쐬���A�ēx���x���`�d�����s���ĕ]�����܂��B
print("\n--- �����_���ȏ������x�����g�p�����ꍇ�̕]�� ---")

random_initial_labels = {}



'''
# �x�z�I�ȃ��x���̏������x�����Ɠ����������A�����_���Ƀm�[�h��I��
num_initial_nodes = len(initial_labels) # �N���[�N����I�΂ꂽ�������x�����Ɠ�����
if len(nodes) >= num_initial_nodes:
    random_initial_nodes = random.sample(nodes, num_initial_nodes)
    # �I�����ꂽ�m�[�h�ɁA���̃m�[�h�̃m�C�Y������̃��x�����������x���Ƃ��Đݒ�
    for node in random_initial_nodes:
        if node in G.nodes(): # �O���t�ɑ��݂���m�[�h���m�F
            random_initial_labels[node] = G.nodes[node]['label'] # �m�C�Y������̃��x�����g�p
else:
    print("�x��: �m�[�h������菉�����x�����̕����������߁A�����_���ȏ������x���̑I�����X�L�b�v���܂��B")
    random_initial_labels = {} # ��̎�����ݒ�
'''

# �����_���ȏ������x�����g�p�����ꍇ�̕]��
# �V���Ƀ����_���ȏ������x���Z�b�g���쐬���A�ēx���x���`�d�����s���ĕ]�����܂��B
print("\n--- �����_���ȏ������x�����g�p�����ꍇ�̕]�� ---")

random_initial_labels = {}
# �������x���̃p�[�Z���e�[�W��ݒ�
random_initial_percentage = 0.95 # ��������: 10%�̃m�[�h�������_���ɑI��

# �����_���ɑI�����鏉�����x���̐����v�Z
num_random_initial_nodes = int(len(nodes) * random_initial_percentage)

# �I�����ꂽ�m�[�h�ɁA���̃m�[�h�̃m�C�Y������̃��x�����������x���Ƃ��Đݒ�
if len(nodes) >= num_random_initial_nodes:
    random_initial_nodes = random.sample(nodes, num_random_initial_nodes)
    for node in random_initial_nodes:
        if node in G.nodes(): # �O���t�ɑ��݂���m�[�h���m�F
            random_initial_labels[node] = G.nodes[node]['label'] # �m�C�Y������̃��x�����g�p
else:
    print("�x��: �m�[�h������菉�����x�����̕����������߁A�����_���ȏ������x���̑I�����X�L�b�v���܂��B")
    random_initial_labels = {} # ��̎�����ݒ�



if random_initial_labels:
    print(f"�����_���ȏ������x���Ƃ��� {len(random_initial_labels)} �̃m�[�h��I�����܂����B")

    # �����_���ȏ������x���Ń��x���`�d�����s
    random_propagated_labels_dict = label_spreading(G, random_initial_labels)

    # �����_���ȏ������x�����g�p�����ꍇ�� TP, TN, FP, FN ���v�Z
    random_TP = 0
    random_FN = 0
    random_FP = 0
    random_TN = 0

    # changed_nodes (�m�C�Y�������ꂽ�m�[�h) �ɂ��ĕ]��
    for node in changed_nodes:
        actual_original_label = original_labels_dict[node] # �m�C�Y�����O�̐^�̃��x��
        predicted_propagated_label = random_propagated_labels_dict.get(node) # �����_���������x���ł̓`�d��̃��x��

        if predicted_propagated_label is not None:
            if predicted_propagated_label == actual_original_label:
                random_TP += 1
            else:
                random_FN += 1
        # else:
        #     random_FN += 1

    # changed_nodes �ȊO�̃m�[�h (�m�C�Y��������Ă��Ȃ��m�[�h) �ɂ��ĕ]��
    for node in unchanged_nodes:
        actual_original_label = original_labels_dict[node] # �m�C�Y�����O�̐^�̃��x��
        predicted_propagated_label = random_propagated_labels_dict.get(node) # �����_���������x���ł̓`�d��̃��x��

        if predicted_propagated_label is not None:
            if predicted_propagated_label == actual_original_label:
                random_TN += 1
            else:
                random_FP += 1
        # else:
        #     pass


    print("\n�����_���ȏ������x���ł̃m�C�Y�C���̕]��:")
    print("TP (�m�C�Y�C������):", random_TP)
    print("FN (�m�C�Y�C�����s):", random_FN)
    print("FP (����ă��x���ύX):", random_FP)
    print("TN (���������x���ێ�):", random_TN)

    random_total_evaluated_nodes = random_TP + random_FN + random_FP + random_TN
    print('���v�]���m�[�h���F', random_total_evaluated_nodes)


    # �K�����A�Č����A���x�AF�l���v�Z�i�����_���ȏ������x�����g�p�j
    random_precision = random_TP / (random_TP + random_FP) if (random_TP + random_FP) > 0 else 0
    random_recall = random_TP / (random_TP + random_FN) if (random_TP + random_FN) > 0 else 0
    random_accuracy = (random_TP + random_TN) / (random_TP + random_TN + random_FP + random_FN) if (random_TP + random_TN + random_FP + random_FN) > 0 else 0
    random_f1 = 2 * (random_precision * random_recall) / (random_precision + random_recall) if (random_precision + random_recall) > 0 else 0

    print("\n�����_���ȏ������x���ł̃m�C�Y�C���̕]���w�W:")
    print(f"�K���� (Precision): {random_precision:.4f}")
    print(f"�Č��� (Recall): {random_recall:.4f}")
    print(f"���x (Accuracy): {random_accuracy:.4f}")
    print(f"F�l (F1-score): {random_f1:.4f}")

else:
    print("\n�����_���ȏ������x�����I������Ȃ��������߁A�]���̓X�L�b�v����܂��B")


# (�I�v�V����) ����
# �����ʂ�2�����̏ꍇ�̂݉���
# ������Ԃł̉����̓��x���`�d�̉ߒ��𒼐ڎ������̂ł͂Ȃ����߁A�����ł̓X�L�b�v���܂��B
# �K�v�ł���΁A�m�[�h�̐F�����̃��x���A�m�C�Y������̃��x���A�`�d��̃��x���œh�蕪����Ȃǂ̉�����ǉ��ł��܂��B
# ��Fnx.draw(G, with_labels=True, node_color=[G.nodes[n]['propagated_label'] for n in G.nodes()]) �Ȃ�

