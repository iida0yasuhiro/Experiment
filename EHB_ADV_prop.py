# �������@��Ď�@�B�m�����z�̂܂܃��x���`�d�ɃC���v�b�g
# �����Q�@�{�����@11�l�̍����s��ɂ��␳����E�Ȃ��Ŕ�r�B�ǂ�����m�����z�̂܂�
# K=1 35.41, 33.80, 33.54 (�␳�Ȃ�)
# K=1 34.77, 32.49, 34.27 (�␳����)
# K=2 50.27, 42.25, 53.39 (�␳�Ȃ�)
# K=2 76.90, 71.25, 75.75 (�␳����)
# K=3 79.38, 69.92, 68.88 (�␳�Ȃ�)
# K=3 88.63, 93.01, 89.05 (�␳����)
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import random
import time

# Constants
ALPHA = 0.8
SIMILARITY_THRESHOLD = 0.8
INITIAL_LABEL_RATIO = 0.17
TOP_K = 3
TOLERANCE = 1e-6
MAX_ITER = 100
RANDOM_SEED = None

def build_graph(features, similarity_threshold):
    """
    �����x�N�g������R�T�C���ގ��x�Ɋ�Â��ďd�ݍs��W���\�z����B
    """
    print("�R�T�C���ގ��x���v�Z��...")
    similarity_matrix = cosine_similarity(features)
    print("�d�ݍs��W���\�z��...")
    W = similarity_matrix * (similarity_matrix >= similarity_threshold)
    return W

def get_true_labels(file_names):
    """
    �t�@�C��������^�̃��x���i���l�j���擾����B
    """
    true_label_map = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
    label_mapping = {
        'NILM': 'NILM', 'ASCUS': 'ASC-US', 'LSIL': 'LSIL',
        'ASCH': 'ASC-H', 'HSIL': 'HSIL', 'SCC': 'SCC'
    }
    true_labels = []
    for fname in file_names:
        name_without_ext = fname.split('.')[0]
        prefix = ''.join(filter(str.isalpha, name_without_ext))
        mapped_prefix = label_mapping.get(prefix, prefix)
        true_labels.append(true_label_map[mapped_prefix])
    return np.array(true_labels)

def label_propagation(Y0, W, max_iter, tolerance):
    """
    ���x���`�d�A���S���Y�������s����B
    """
    print("���x���`�d���J�n���܂�...")
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-12))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    Y = Y0.copy()
    for i in range(max_iter):
        Y_prev = Y.copy()
        Y = ALPHA * S @ Y + (1 - ALPHA) * Y0
        if np.linalg.norm(Y - Y_prev) < tolerance:
            print(f"�������܂����B������: {i+1}")
            break
        if (i + 1) % 10 == 0:
            print(f"������: {i+1}/{max_iter}")
    else:
        print(f"�ő唽���񐔂ɒB���܂����B������: {max_iter}")
    sum_Y = Y.sum(axis=1, keepdims=True)
    Y = np.divide(Y, sum_Y, out=np.zeros_like(Y), where=sum_Y != 0)
    return Y

def apply_confusion_matrix_correction(Y_final, confusion_matrix):
    """
    �A�m�e�[�^�̍����s����g���āA���x���`�d�̍ŏI���ʂ�␳����
    """
    print("�����s��ɂ�郉�x���X�R�A�̕␳���J�n���܂�...")
    conf_matrix_prob = confusion_matrix / 100
    Y_updated = np.dot(Y_final, conf_matrix_prob)
    Y_updated = Y_updated / Y_updated.sum(axis=1, keepdims=True)
    print("�␳���������܂����B")
    return Y_updated

def write_results(filename, num_nodes, file_names, true_labels, Y_result, unselected_fnames, title):
    """
    ���ʂ��t�@�C���ɏ������ދ��ʊ֐�
    """
    true_label_map = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
    true_label_map_rev = {v: k for k, v in true_label_map.items()}
    predicted_labels = np.argmax(Y_result, axis=1)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- {title} ---\n")
        f.write(f"���s���ꂽ�m�[�h��: {num_nodes}\n\n")
        f.write(f"{'�t�@�C����':<20}{'�^�̃��x��':<15}{'�\�����x��':<15}{'�ŏI���x�����z�x�N�g�� (�m��)':<}\n")
        f.write("-" * 100 + "\n")
        for i, fname in enumerate(file_names):
            true_label_name = true_label_map_rev.get(true_labels[i], 'Unknown')
            predicted_label_name = true_label_map_rev.get(predicted_labels[i], 'Unknown')
            f.write(f"{fname:<20}{true_label_name:<15}{predicted_label_name:<15}{np.round(Y_result[i], 4)}\n")

    print(f"���ʂ� '{filename}' �ɏ������݂܂����B4�b�ԑҋ@���܂�...")
    time.sleep(4)
    print("�ҋ@�����B")

    with open(filename, "a", encoding="utf-8") as f:
        correct_predictions = 0
        total_unselected_nodes = 0
        unselected_results_output = []
        eval_fnames = [fname for fname in file_names if fname in unselected_fnames]
        for fname in eval_fnames:
            i = file_names.index(fname)
            total_unselected_nodes += 1
            final_distribution = Y_result[i]
            true_class = true_labels[i]
            predicted_top_k_indices = np.argsort(final_distribution)[-TOP_K:]
            is_correct = true_class in predicted_top_k_indices
            if is_correct:
                correct_predictions += 1
            unselected_results_output.append({
                "fname": fname,
                "true_label": true_label_map_rev.get(true_class, 'Unknown'),
                "predicted_top_k_labels": [true_label_map_rev.get(idx, 'Unknown') for idx in predicted_top_k_indices],
                "probability_vector": np.round(final_distribution, 4).tolist(),
            })
        accuracy = (correct_predictions / total_unselected_nodes) * 100 if total_unselected_nodes > 0 else 0
        f.write("\n" + "=" * 50 + "\n")
        f.write("--- �I�����Ȃ������m�[�h�̗\�����x (�g�b�vK) ---\n")
        f.write(f"�g�b�vK�̐�: {TOP_K}\n")
        f.write(f"�]���Ώۃm�[�h��: {total_unselected_nodes}\n")
        f.write(f"�����m�[�h��: {correct_predictions}\n")
        f.write(f"���x: {accuracy:.2f}%\n")
        f.write("\n--- �I�����Ȃ������m�[�h�̏ڍ׌��� ---\n")
        for result in unselected_results_output:
            f.write(f"�t�@�C����: {result['fname']}\n")
            f.write(f"  �^�̃��x��: {result['true_label']}\n")
            f.write(f"  �\���g�b�vK���x��: {result['predicted_top_k_labels']}\n")
            f.write(f"  �ŏI�m���x�N�g��: {result['probability_vector']}\n")
            f.write("-" * 50 + "\n")
    print(f"���ʂ� '{filename}' �ɏ������܂�܂����B")
    print(f"�I�����Ȃ������m�[�h�̗\�����x: {accuracy:.2f}%")

def main():
    random.seed(RANDOM_SEED)

    # --- �f�[�^�̓ǂݍ��� ---
    print("SM-official.json����f�[�^��ǂݍ��ݒ�...")
    try:
        with open('SM-official.json', 'r', encoding='utf-8') as f:
            sm_official_data_json = json.load(f)
    except FileNotFoundError:
        print("�G���[: 'SM-official.json'�t�@�C����������܂���B")
        return
    sm_official_data = {}
    for key, value in sm_official_data_json.items():
        new_key = key.replace('./', '')
        sm_official_data[new_key] = value
    file_names = list(sm_official_data.keys())
    features = np.array(list(sm_official_data.values()))
    num_nodes = len(file_names)
    print(f"�m�[�h��: {num_nodes}")

    # error_distribution_vectors.txt�̓ǂݍ���
    all_initial_labels = {}
    with open('error_distribution_vectors.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("�m�[�h�i�t�@�C���j:"):
                fname_with_hyphen = lines[i].split(': ')[1].strip()
                fname_without_hyphen = fname_with_hyphen.replace('-', '') + '.jpg'
                vector_str = lines[i+1].split('[')[1].split(']')[0]
                vector = np.fromstring(vector_str, sep=' ')
                all_initial_labels[fname_without_hyphen] = vector

    # --- �����s��̒�` ---
    confusion_matrix_from_image = np.array([
        [67.5, 23.4, 8.4, 0.6, 0.0, 0.0],
        [1.4, 15.8, 19.2, 13.7, 37.7, 12.3],
        [0.7, 14.4, 41.8, 11.6, 16.4, 15.1],
        [0.0, 5.0, 6.3, 13.2, 49.1, 26.4],
        [2.0, 8.4, 10.1, 14.5, 48.8, 16.2],
        [0.8, 6.7, 5.1, 16.2, 39.9, 31.2]
    ])

    # --- 15-50%�̃m�[�h�������_���ɑI�� ---
    initial_label_fnames = list(all_initial_labels.keys())
    random.shuffle(initial_label_fnames)
    num_initial_labels = int(len(initial_label_fnames) * INITIAL_LABEL_RATIO)
    selected_initial_fnames = set(initial_label_fnames[:num_initial_labels])
    unselected_fnames = set(initial_label_fnames[num_initial_labels:])

    # --- �O���t�̍\�z ---
    W = build_graph(features, SIMILARITY_THRESHOLD)

    # --- �������x���s�� Y0 �̍쐬�i����6�����x�N�g�����g�p�j ---
    Y0 = np.zeros((num_nodes, 6))
    for i, fname in enumerate(file_names):
        if fname in selected_initial_fnames:
            # �ύX�_: one-hot�x�N�g���ł͂Ȃ��A���̃x�N�g�������̂܂܎g�p
            Y0[i] = all_initial_labels[fname]

    # ���K��
    sum_Y0 = Y0.sum(axis=1, keepdims=True)
    Y0 = np.divide(Y0, sum_Y0, out=np.zeros_like(Y0), where=sum_Y0 != 0)

    # --- ���x���`�d�̎��s ---
    Y_final = label_propagation(Y0, W, MAX_ITER, TOLERANCE)

    # --- ���ʂ̕]���Əo�́i�␳�Ȃ��j ---
    true_labels = get_true_labels(file_names)
    write_results("results_no_correction.txt", num_nodes, file_names, true_labels, Y_final, unselected_fnames, "�����s��ɂ��␳�Ȃ��̍ŏI����")

    # --- ��ăA���S���Y���̓K�p ---
    Y_corrected = apply_confusion_matrix_correction(Y_final, confusion_matrix_from_image)

    # --- ���ʂ̕]���Əo�́i�␳����j ---
    write_results("results_with_correction.txt", num_nodes, file_names, true_labels, Y_corrected, unselected_fnames, "�����s��ɂ��␳����̍ŏI����")

if __name__ == "__main__":
    main()