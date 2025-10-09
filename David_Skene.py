# 9ŒŽ23“ú David SkeneƒAƒ‹ƒSƒŠƒYƒ€‚ÌŽÀ‘•
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Label and integer ID mapping
LABEL_MAPPING = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
# Create reverse mapping as well
REV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
NUM_CLASSES = len(LABEL_MAPPING)

def load_all_items(official_file_path):
    """
    Loads all item IDs to be evaluated from the official JSON file.
    Per instruction, it generates the ID by removing the prefix './' and suffix '.jpg' from the JSON keys.
    """
    with open(official_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_items = []
    for key in data.keys():
        # Extract item_id (e.g., 'NILM0005') from key (e.g., './NILM0005.jpg')
        item_id = key.removeprefix('./').removesuffix('.jpg')
        all_items.append(item_id)

    return sorted(list(set(all_items)))

def load_all_annotations(file_paths):
    """
    Loads multiple annotation files (JSON format) and consolidates them
    """
    all_annotations = []
    for file_path in file_paths:
        path = Path(file_path)
        annotator_id = path.stem.split(' ')[1]

        with open(path, 'r', encoding='utf-8') as f:
            try:
                # Replace single quotes to handle files using them
                content = f.read().replace("'", '"')
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: The JSON format of {file_path} might be invalid.")
                continue

            for _, items in data.items():
                for item_id, label in items.items():
                    all_annotations.append({
                        'item_id': item_id,
                        'annotator_id': annotator_id,
                        'label': label
                    })

    return pd.DataFrame(all_annotations)

def initialize_parameters(data_df, all_items):
    """
    Initializes the parameters for the Dawid-Skene algorithm.
    """
    # Initialize only with annotations that exist in the official list
    filtered_df = data_df[data_df['item_id'].isin(all_items)].copy()

    annotators = filtered_df['annotator_id'].unique()
    num_items = len(all_items)

    item_labels_mv = filtered_df.groupby('item_id')['label'].agg(lambda x: x.value_counts().index[0])

    class_counts = item_labels_mv.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    class_priors = (class_counts.values + 1) / (num_items + NUM_CLASSES)

    annotator_errors = {k: np.zeros((NUM_CLASSES, NUM_CLASSES)) for k in annotators}
    item_true_labels_map = item_labels_mv.to_dict()

    for row in filtered_df.itertuples():
        true_label = item_true_labels_map.get(row.item_id)
        if true_label is not None:
            annotator_errors[row.annotator_id][int(true_label), int(row.label)] += 1

    for k in annotators:
        annotator_errors[k] += 1
        row_sums = annotator_errors[k].sum(axis=1, keepdims=True)
        annotator_errors[k] = np.divide(annotator_errors[k], row_sums,
                                        out=np.full_like(annotator_errors[k], 1/NUM_CLASSES),
                                        where=row_sums!=0)

    return class_priors, annotator_errors

def dawid_skene(data_df, all_items, max_iter=30, tol=1e-5):
    """
    Runs the Dawid-Skene algorithm (EM algorithm).
    """
    annotators = data_df['annotator_id'].unique()
    num_items = len(all_items)

    class_priors, annotator_errors = initialize_parameters(data_df, all_items)

    data_map = {item: {} for item in all_items}

    # Check if items in annotation files exist in the official list (all_items)
    ignored_items = set()
    for row in data_df.itertuples():
        if row.item_id in data_map:
            data_map[row.item_id][row.annotator_id] = row.label
        elif row.item_id not in ignored_items:
            # Display a warning only once for items that don't exist
            print(f"Warning: Item '{row.item_id}' does not exist in 'SM-official.json', so its annotations will be ignored.")
            ignored_items.add(row.item_id)

    old_log_likelihood = -np.inf

    for i in range(max_iter):
        item_probas = pd.DataFrame(index=all_items, columns=range(NUM_CLASSES), dtype=float)

        for item_id in all_items:
            log_probas = np.log(class_priors.copy())

            if item_id in data_map and data_map[item_id]:
                for j in range(NUM_CLASSES):
                    for annotator_id, label in data_map[item_id].items():
                        error_matrix = annotator_errors[annotator_id]
                        if error_matrix[j, label] > 1e-9:
                            log_probas[j] += np.log(error_matrix[j, label])

            max_log = np.max(log_probas)
            probas_exp = np.exp(log_probas - max_log)
            item_probas.loc[item_id] = probas_exp / np.sum(probas_exp)

        class_priors = item_probas.sum(axis=0).values / num_items

        for annotator_id in annotators:
            error_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
            annotator_data = data_df[(data_df['annotator_id'] == annotator_id) & (data_df['item_id'].isin(all_items))]

            for row in annotator_data.itertuples():
                error_matrix[:, row.label] += item_probas.loc[row.item_id].values

            annotated_items = annotator_data['item_id'].unique()
            if len(annotated_items) > 0:
                sum_T_ij = item_probas.loc[annotated_items].sum(axis=0).values
                annotator_errors[annotator_id] = np.divide(error_matrix, sum_T_ij[:, np.newaxis],
                                                        out=np.full_like(error_matrix, 1/NUM_CLASSES),
                                                        where=sum_T_ij[:, np.newaxis] > 1e-9)

        log_likelihood = 0.0
        for item_id in all_items:
            item_ll_terms = class_priors.copy()
            if item_id in data_map:
                for j in range(NUM_CLASSES):
                    for annotator_id, label in data_map[item_id].items():
                        item_ll_terms[j] *= annotator_errors[annotator_id][j, label]

            total_prob = np.sum(item_ll_terms)
            if total_prob > 1e-9:
                log_likelihood += np.log(total_prob)

        print(f"Iteration {i+1}: Log-Likelihood = {log_likelihood:.4f}")
        if i > 0 and abs(log_likelihood - old_log_likelihood) < tol:
            print("Converged.")
            break
        old_log_likelihood = log_likelihood

    item_probas.columns = [REV_LABEL_MAPPING[c] for c in item_probas.columns]

    return {
        'item_probas': item_probas,
        'annotator_errors': annotator_errors,
        'class_priors': class_priors
    }

def main():
    """
    Main execution function
    """
    annotator_files = [
        '01 AK0326.txt', '02 HA0523.txt', '03 MI0722.txt', '04 SI0417.txt',
        '05 S1017N.txt', '06 SS0122.txt', '07 SS19750144.txt', '08 TK0317.txt',
        '09 TM0725.txt', '10 TM0815.txt', '11 YH0309.txt'
    ]
    official_file = 'SM-official.json'

    try:
        all_items_list = load_all_items(official_file)
        annotations_df = load_all_annotations(annotator_files)
        print("? All files loaded successfully.")
        print(f" ?- Total items (from SM-official.json): {len(all_items_list)}")
        print(f" ?- Total annotations: {len(annotations_df)}")
        print(f" ?- Number of annotators: {annotations_df['annotator_id'].nunique()}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the necessary files are in the correct directory.")
        return

    print("\nStarting the Dawid-Skene algorithm...")
    results = dawid_skene(annotations_df, all_items_list)

    estimated_true_labels = results['item_probas'].idxmax(axis=1)

    print("\n--- š Estimated True Labels (first 20) ---")
    print(estimated_true_labels.head(20))

    estimated_true_labels.to_csv('estimated_true_labels.csv', header=['estimated_label'], index_label='item_id')
    print("\n? Estimated labels for all items have been saved to 'estimated_true_labels.csv'.")

    print("\n--- š Final Class Prior Probabilities ---")
    for i, prior in enumerate(results['class_priors']):
        print(f" ?- {REV_LABEL_MAPPING[i]:<6}: {prior:.4f}")

    print("\n--- š Estimated Confusion Matrix for Each Annotator ---")
    print("(Rows: True Label, Columns: Label assigned by Annotator)")
    for annotator_id, matrix in results['annotator_errors'].items():
        print(f"\nAnnotator: {annotator_id}")
        matrix_df = pd.DataFrame(matrix,
                                 index=[f"True_{REV_LABEL_MAPPING[i]}" for i in range(NUM_CLASSES)],
                                 columns=[f"Ann_{REV_LABEL_MAPPING[i]}" for i in range(NUM_CLASSES)])
        print(matrix_df.round(3))

    # Add the new evaluation code below

    # Extract the prefix (non-numeric part) from the item_id
    item_prefixes = estimated_true_labels.index.str.replace(r'\d+', '', regex=True)

    # Compare whether the estimated label matches the prefix
    matches = (item_prefixes == estimated_true_labels)

    # Calculate the number of matches and total number of items
    total_items = len(estimated_true_labels)
    matched_count = matches.sum()

    # Output the final result
    print("\n--- š Match Rate between Estimated True Labels and Item Prefixes ---")
    print(f" ?- Total number of items: {total_items}")
    print(f" ?- Number of matched items: {matched_count}")
    if total_items > 0:
        print(f" ?- Match Rate: {matched_count / total_items:.2%}")
    else:
        print(" ?- Match Rate: 0.00%")

if __name__ == '__main__':
    main()