# ��Sample-wise Label Fusion 10��3������ ���Ԃ�����΂ǂ����Ō������Ă���
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import re

# --- �萔��` ---
LABEL_MAPPING = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
REV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
NUM_CLASSES = len(LABEL_MAPPING)
FEATURE_DIM = 768
EPSILON = 1e-12 # ���l���艻�̂��߂̋ɏ��l

# --- �t�@�C���ǂݍ��݊֐� ---
def load_all_items(official_file_path):
    with open(official_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_items = [key.removeprefix('./').removesuffix('.jpg') for key in data.keys()]
    return sorted(list(set(all_items)))

def load_all_annotations(file_paths):
    """
    ���K�\����p���āA�������𖳎����A�A�m�e�[�V�����f�[�^�݂̂𒊏o����
    """
    all_annotations = []
    # ���K�\���p�^�[��: "ITEM_ID": LABEL �܂��� 'ITEM_ID': LABEL �̌`���Ɉ�v
    pattern = re.compile(r'["\']([A-Z]+-?[A-Z]*\d+)["\']\s*:\s*(\d+)')

    for file_path in file_paths:
        path = Path(file_path)
        try:
            annotator_id = path.stem.split(' ')[1]
        except IndexError:
            print(f"�t�@�C���� '{path.name}' �̃G���[")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = pattern.findall(content)

            if not matches:
                print(f" {file_path} ����A�m�e�[�V�����f�[�^���o���s�B")
                continue

            for item_id, label_str in matches:
                label_int = int(label_str)
                # LABEL_MAPPING�ɑ��݂��郉�x���N���X���m�F
                if label_int in REV_LABEL_MAPPING:
                    all_annotations.append({
                        'item_id': item_id,
                        'annotator_id': annotator_id,
                        'label': label_int
                    })

    df = pd.DataFrame(all_annotations)
    df = df.drop_duplicates(subset=['item_id', 'annotator_id'], keep='first')
    return df

def load_features(official_file_path, all_items):
    with open(official_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    feature_dict = {
        key.removeprefix('./').removesuffix('.jpg'): value
        for key, value in data.items()
    }

    features_df = pd.DataFrame.from_dict(feature_dict, orient='index')
    features_df = features_df.reindex(all_items)

    if features_df.isnull().values.any():
        print("�����x�N�g�������݂��Ȃ��ꍇ��0�ŕ⊮�B")
        features_df = features_df.fillna(0)

    return features_df

# --- PyTorch�p�f�[�^�Z�b�g ---
class NoisyLabelsDataset(Dataset):
    def __init__(self, features, noisy_labels):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.LongTensor(noisy_labels.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- �j���[�����l�b�g�������� ---
class LabelFusionNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_annotators, num_basis_matrices, hidden_dim=128):
        super(LabelFusionNet, self).__init__()
        self.num_annotators = num_annotators
        self.num_basis_matrices = num_basis_matrices

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.prediction_head = nn.Linear(hidden_dim, num_classes)
        self.weights_head = nn.Linear(hidden_dim, num_annotators)
        self.coeffs_head = nn.Linear(hidden_dim, num_annotators * num_basis_matrices)

    def forward(self, x):
        shared_features = self.backbone(x)
        class_logits = self.prediction_head(shared_features)
        annotator_weights = torch.softmax(self.weights_head(shared_features), dim=1)

        confusion_coeffs = self.coeffs_head(shared_features).view(
            -1, self.num_annotators, self.num_basis_matrices
        )
        confusion_coeffs = torch.softmax(confusion_coeffs, dim=2)
        return class_logits, annotator_weights, confusion_coeffs

# --- ���f���̊w�K�Ɛ��_���s�����C���N���X ---
class SampleWiseLabelFusion:
    def __init__(self, all_items, annotations_df, features_df, hyperparams):
        self.hyperparams = hyperparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"�g�p�f�o�C�X: {self.device}")

        annotator_list = sorted(annotations_df['annotator_id'].unique())
        self.annotator_map = {name: i for i, name in enumerate(annotator_list)}
        self.num_annotators = len(self.annotator_map)

        annotations_df['annotator_idx'] = annotations_df['annotator_id'].map(self.annotator_map)

        # -1�̓A�m�e�[�V�����Ȃ��̈Ӗ�
        pivoted_labels = annotations_df.pivot(index='item_id', columns='annotator_idx', values='label')
        self.noisy_labels_df = pivoted_labels.reindex(all_items).fillna(-1)
        self.features_df = features_df.reindex(all_items)

        self.model = LabelFusionNet(
            input_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            num_annotators=self.num_annotators,
            num_basis_matrices=self.hyperparams['M']
        ).to(self.device)

        # �w�K����0.0001�ɉ����Ă�������
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

        self.basis_matrices = self._generate_permutation_matrices(
            self.hyperparams['M'], NUM_CLASSES
        ).to(self.device)

    def _generate_permutation_matrices(self, M, K):
        matrices = torch.zeros(M, K, K)
        identity = torch.eye(K)
        for i in range(M):
            # ����s��𐶐�
            matrices[i] = identity[torch.randperm(K)]
        return matrices

    def _compute_loss(self, logits, weights, coeffs, noisy_labels):
        P = torch.einsum('brm,mkl->brkl', coeffs, self.basis_matrices)

        # -1 (�A�m�e�[�V�����Ȃ�) �̃��x���𖳎����邽�߁Aone-hot
        y_noisy_onehot = F.one_hot(noisy_labels.clamp(min=0), num_classes=NUM_CLASSES).float()

        # �m�C�Y�s�� P ��K�p���āA�A�m�e�[�^ r ���^�̃��x�� k �� c �ƌ땪�ނ���m���̃x�N�g���𓾂�
        y_clean = torch.matmul(P, y_noisy_onehot.unsqueeze(-1)).squeeze(-1)

        # �A�m�e�[�V���������݂���ӏ��̂݃}�X�N
        mask = (noisy_labels != -1).float().unsqueeze(-1)

        # �A�m�e�[�^�̏d�݂ƃ}�X�N��K�p���ă\�t�g�^�[�Q�b�g���v�Z
        y_targ = torch.sum(weights.unsqueeze(-1) * y_clean * mask, dim=1)

        # ������ ���l���艻�̂��߂̏C��: �[�����Z�΍� ������
        y_targ = y_targ + EPSILON
        y_targ_sum = y_targ.sum(dim=1, keepdim=True)
        y_targ = y_targ / y_targ_sum # ���K��

        log_preds = F.log_softmax(logits, dim=1)

        # KL�_�C�o�[�W�F���X����
        # y_targ��EPSILON�������Ă��邽�߁ALog(0)�ɂ��nan�������X�N���啝�Ɍ���
        loss_kl = F.kl_div(log_preds, y_targ, reduction='batchmean')

        # �������� (�m�C�Y�s�� P �̑Ίp�v�f��1�ɋ߂��Ȃ�悤�ɗU�������Ă݂�)
        diag_P = torch.diagonal(P, dim1=-2, dim2=-1)
        reg_term = torch.sum((1.0 - diag_P)**2, dim=2)
        loss_reg = torch.mean(torch.sum(reg_term * mask.squeeze(-1), dim=1) / (mask.sum(dim=1).clamp(min=EPSILON)))

        return loss_kl + self.hyperparams['lambda'] * loss_reg

    def train(self):
        dataset = NoisyLabelsDataset(self.features_df, self.noisy_labels_df)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

        self.model.train()
        print("���f���w�K�J�n...")
        for epoch in range(self.hyperparams['epochs']):
            total_loss = 0
            for features, noisy_labels in tqdm(dataloader, desc=f"�G�|�b�N {epoch+1}/{self.hyperparams['epochs']}"):
                features, noisy_labels = features.to(self.device), noisy_labels.to(self.device)

                self.optimizer.zero_grad()
                logits, weights, coeffs = self.model(features)
                loss = self._compute_loss(logits, weights, coeffs, noisy_labels)

                # ������nan�̏ꍇ�A�w�K�𒆒f
                if torch.isnan(loss):
                    print(f"\n �G�|�b�N {epoch+1} �ő����� nan �̏ꍇ�w�K���f�B")
                    return

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"�G�|�b�N {epoch+1} | ���ϑ���: {total_loss / len(dataloader):.4f}")

    def predict(self):
        self.model.eval()
        dataset = NoisyLabelsDataset(self.features_df, self.noisy_labels_df)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for features, _ in tqdm(dataloader, desc="�\���v�Z��"):
                features = features.to(self.device)
                logits, _, _ = self.model(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return pd.Series(all_preds, index=self.features_df.index).map(REV_LABEL_MAPPING)

def main():
    annotator_files = [
        '01 AK0326.txt', '02 HA0523.txt', '03 MI0722.txt', '04 SI0417.txt',
        '05 S1017N.txt', '06 SS0122.txt', '07 SS19750144.txt', '08 TK0317.txt',
        '09 TM0725.txt', '10 TM0815.txt', '11 YH0309.txt'
    ]
    official_file = 'SM-official.json'

    try:
        all_items_list = load_all_items(official_file)
        annotations_df = load_all_annotations(annotator_files)
        features_df = load_features(official_file, all_items_list)

        if annotations_df.empty:
            print("\n�A�m�e�[�V�����t�@�C����Read���s�B")
            return

        print("? �S�t�@�C���̓ǂݍ��ݐ����B")
        print(f" ?- �A�C�e������: {len(all_items_list)}")
        print(f" ?- �A�m�e�[�V��������: {len(annotations_df)}")
        print(f" ?- �A�m�e�[�^�[��: {annotations_df['annotator_id'].nunique()}")
    except FileNotFoundError as e:
        print(f"�G���[: {e}�B�t�@�C�����������f�B���N�g���ɑ��݂��邩�B")
        return

    # ������ �w�K���� 0.0001 �ɕύX�����B0.001�ł͎������Ȃ� ������
    hyperparams = {
        'lr': 0.0001,
        'epochs': 20,
        'batch_size': 64,
        'lambda': 1.0,
        'M': 30
    }

    fusion_algorithm = SampleWiseLabelFusion(all_items_list, annotations_df, features_df, hyperparams)
    fusion_algorithm.train()

    # nan�Ŋw�K�����f���ꂽ�ꍇ�͗\���X�L�b�v
    if not (hasattr(fusion_algorithm, 'model') and torch.isnan(fusion_algorithm.model.backbone[0].weight.data).any()):
        estimated_true_labels = fusion_algorithm.predict()

        print("\n--- �� ���肳�ꂽ�^�̃��x�� (�擪20��) ---")
        print(estimated_true_labels.head(20))

        estimated_true_labels.to_csv('estimated_true_labels.csv', header=['estimated_label'], index_label='item_id')
        print("\n? �S�f�[�^�̐��胉�x���� 'estimated_true_labels.csv' �ɕۑ��ρB")

        print("\n--- �� ���胉�x���ƃA�C�e���v���t�B�b�N�X�̈�v�� ---")
        item_prefixes = estimated_true_labels.index.str.replace(r'\d+', '', regex=True)
        matches = (item_prefixes == estimated_true_labels)
        total_items = len(estimated_true_labels)
        matched_count = matches.sum()

        print(f" ?- �A�C�e������: {total_items}")
        print(f" ?- ��v�����A�C�e����: {matched_count}")
        if total_items > 0:
            print(f" ?- ��v��: {matched_count / total_items:.2%}")
        else:
            print(" ?- ��v��: 0.00%")

if __name__ == '__main__':
    main()