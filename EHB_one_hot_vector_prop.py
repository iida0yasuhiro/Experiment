# ★★★　実験２　本実験　９月２２日　11人の混同行列による補正あり・なしで比較。どちらもone-hotベクトルにしたもの
# K=1 32.31, 33.42, 35.04 (補正なし)
# K=1 33.08, 32.89, 34.50 (補正あり)
# K=2 58.54, 49.48, 47.45 (補正なし)
# K=2 79.40, 76.55, 74.53 (補正あり)
# K=3 65.09, 63.54, 70.43 (補正なし)
# K=3 90.03, 91.39, 90.86 (補正あり)
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import random
import time

# Constants
ALPHA = 0.8
SIMILARITY_THRESHOLD = 0.8
INITIAL_LABEL_RATIO = 0.5
TOP_K = 1
TOLERANCE = 1e-6
MAX_ITER = 100
RANDOM_SEED = None

def build_graph(features, similarity_threshold):
    """
    特徴ベクトルからコサイン類似度に基づいて重み行列Wを構築する。
    """
    print("コサイン類似度を計算中...")
    similarity_matrix = cosine_similarity(features)
    print("重み行列Wを構築中...")
    W = similarity_matrix * (similarity_matrix >= similarity_threshold)
    return W

def get_true_labels(file_names):
    """
    ファイル名から真のラベル（数値）を取得する。
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
    ラベル伝播アルゴリズムを実行する。
    """
    print("ラベル伝播を開始します...")
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-12))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    Y = Y0.copy()
    for i in range(max_iter):
        Y_prev = Y.copy()
        Y = ALPHA * S @ Y + (1 - ALPHA) * Y0
        if np.linalg.norm(Y - Y_prev) < tolerance:
            print(f"収束しました。反復回数: {i+1}")
            break
        if (i + 1) % 10 == 0:
            print(f"反復回数: {i+1}/{max_iter}")
    else:
        print(f"最大反復回数に達しました。反復回数: {max_iter}")
    sum_Y = Y.sum(axis=1, keepdims=True)
    Y = np.divide(Y, sum_Y, out=np.zeros_like(Y), where=sum_Y != 0)
    return Y

def apply_confusion_matrix_correction(Y_final, confusion_matrix):
    """
    アノテータの混同行列を使って、ラベル伝播の最終結果を補正する
    """
    print("混同行列によるラベルスコアの補正を開始します...")
    conf_matrix_prob = confusion_matrix / 100
    Y_updated = np.dot(Y_final, conf_matrix_prob)
    Y_updated = Y_updated / Y_updated.sum(axis=1, keepdims=True)
    print("補正が完了しました。")
    return Y_updated

def write_results(filename, num_nodes, file_names, true_labels, Y_result, unselected_fnames, title):
    """
    結果をファイルに書き込む共通関数
    """
    true_label_map = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
    true_label_map_rev = {v: k for k, v in true_label_map.items()}
    predicted_labels = np.argmax(Y_result, axis=1)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- {title} ---\n")
        f.write(f"実行されたノード数: {num_nodes}\n\n")
        f.write(f"{'ファイル名':<20}{'真のラベル':<15}{'予測ラベル':<15}{'最終ラベル分布ベクトル (確率)':<}\n")
        f.write("-" * 100 + "\n")
        for i, fname in enumerate(file_names):
            true_label_name = true_label_map_rev.get(true_labels[i], 'Unknown')
            predicted_label_name = true_label_map_rev.get(predicted_labels[i], 'Unknown')
            f.write(f"{fname:<20}{true_label_name:<15}{predicted_label_name:<15}{np.round(Y_result[i], 4)}\n")

    print(f"結果を '{filename}' に書き込みました。4秒間待機します...")
    time.sleep(4)
    print("待機完了。")

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
        f.write("--- 選択しなかったノードの予測精度 (トップK) ---\n")
        f.write(f"トップKの数: {TOP_K}\n")
        f.write(f"評価対象ノード数: {total_unselected_nodes}\n")
        f.write(f"正解ノード数: {correct_predictions}\n")
        f.write(f"精度: {accuracy:.2f}%\n")
        f.write("\n--- 選択しなかったノードの詳細結果 ---\n")
        for result in unselected_results_output:
            f.write(f"ファイル名: {result['fname']}\n")
            f.write(f"  真のラベル: {result['true_label']}\n")
            f.write(f"  予測トップKラベル: {result['predicted_top_k_labels']}\n")
            f.write(f"  最終確率ベクトル: {result['probability_vector']}\n")
            f.write("-" * 50 + "\n")
    print(f"結果が '{filename}' に書き込まれました。")
    print(f"選択しなかったノードの予測精度: {accuracy:.2f}%")

def main():
    random.seed(RANDOM_SEED)

    # --- データの読み込み ---
    print("SM-official.jsonからデータを読み込み中...")
    try:
        with open('SM-official.json', 'r', encoding='utf-8') as f:
            sm_official_data_json = json.load(f)
    except FileNotFoundError:
        print("エラー: 'SM-official.json'ファイルが見つかりません。")
        return
    sm_official_data = {}
    for key, value in sm_official_data_json.items():
        new_key = key.replace('./', '')
        sm_official_data[new_key] = value
    file_names = list(sm_official_data.keys())
    features = np.array(list(sm_official_data.values()))
    num_nodes = len(file_names)
    print(f"ノード数: {num_nodes}")

    # error_distribution_vectors.txtの読み込み
    all_initial_labels = {}
    with open('error_distribution_vectors.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("ノード（ファイル）:"):
                fname_with_hyphen = lines[i].split(': ')[1].strip()
                fname_without_hyphen = fname_with_hyphen.replace('-', '') + '.jpg'
                vector_str = lines[i+1].split('[')[1].split(']')[0]
                vector = np.fromstring(vector_str, sep=' ')
                all_initial_labels[fname_without_hyphen] = vector

    # --- 混同行列の定義 ---
    confusion_matrix_from_image = np.array([
        [67.5, 23.4, 8.4, 0.6, 0.0, 0.0],
        [1.4, 15.8, 19.2, 13.7, 37.7, 12.3],
        [0.7, 14.4, 41.8, 11.6, 16.4, 15.1],
        [0.0, 5.0, 6.3, 13.2, 49.1, 26.4],
        [2.0, 8.4, 10.1, 14.5, 48.8, 16.2],
        [0.8, 6.7, 5.1, 16.2, 39.9, 31.2]
    ])

    # --- 15から50%のノードをランダムに選択 ---
    initial_label_fnames = list(all_initial_labels.keys())
    random.shuffle(initial_label_fnames)
    num_initial_labels = int(len(initial_label_fnames) * INITIAL_LABEL_RATIO)
    selected_initial_fnames = set(initial_label_fnames[:num_initial_labels])
    unselected_fnames = set(initial_label_fnames[num_initial_labels:])

    # --- グラフの構築 ---
    W = build_graph(features, SIMILARITY_THRESHOLD)

    # --- 初期ラベル行列 Y0 の作成（one-hotベクトル） ---
    Y0 = np.zeros((num_nodes, 6))
    for i, fname in enumerate(file_names):
        if fname in selected_initial_fnames:
            original_vector = all_initial_labels[fname]
            max_index = np.argmax(original_vector)
            one_hot_vector = np.zeros(6)
            one_hot_vector[max_index] = 1.0
            Y0[i] = one_hot_vector

    # --- ラベル伝播の実行 ---
    Y_final = label_propagation(Y0, W, MAX_ITER, TOLERANCE)

    # --- 結果の評価と出力（補正なし） ---
    true_labels = get_true_labels(file_names)
    write_results("results_no_correction.txt", num_nodes, file_names, true_labels, Y_final, unselected_fnames, "混同行列による補正なしの最終結果")

    # --- 提案アルゴリズムの適用 ---
    Y_corrected = apply_confusion_matrix_correction(Y_final, confusion_matrix_from_image)

    # --- 結果の評価と出力（補正あり） ---
    write_results("results_with_correction.txt", num_nodes, file_names, true_labels, Y_corrected, unselected_fnames, "混同行列による補正ありの最終結果")

if __name__ == "__main__":
    main()