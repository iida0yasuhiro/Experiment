# ★Sample-wise Label Fusion 10月3日実験 時間があればどこかで見直しておく
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

# --- 定数定義 ---
LABEL_MAPPING = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
REV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
NUM_CLASSES = len(LABEL_MAPPING)
FEATURE_DIM = 768
EPSILON = 1e-12 # 数値安定化のための極小値

# --- ファイル読み込み関数 ---
def load_all_items(official_file_path):
    with open(official_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_items = [key.removeprefix('./').removesuffix('.jpg') for key in data.keys()]
    return sorted(list(set(all_items)))

def load_all_annotations(file_paths):
    """
    正規表現を用いて、書式問題を無視し、アノテーションデータのみを抽出する
    """
    all_annotations = []
    # 正規表現パターン: "ITEM_ID": LABEL または 'ITEM_ID': LABEL の形式に一致
    pattern = re.compile(r'["\']([A-Z]+-?[A-Z]*\d+)["\']\s*:\s*(\d+)')

    for file_path in file_paths:
        path = Path(file_path)
        try:
            annotator_id = path.stem.split(' ')[1]
        except IndexError:
            print(f"ファイル名 '{path.name}' のエラー")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = pattern.findall(content)

            if not matches:
                print(f" {file_path} からアノテーションデータ抽出失敗。")
                continue

            for item_id, label_str in matches:
                label_int = int(label_str)
                # LABEL_MAPPINGに存在するラベルクラスか確認
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
        print("特徴ベクトルが存在しない場合に0で補完。")
        features_df = features_df.fillna(0)

    return features_df

# --- PyTorch用データセット ---
class NoisyLabelsDataset(Dataset):
    def __init__(self, features, noisy_labels):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.LongTensor(noisy_labels.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- ニューラルネットここから ---
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

# --- モデルの学習と推論を行うメインクラス ---
class SampleWiseLabelFusion:
    def __init__(self, all_items, annotations_df, features_df, hyperparams):
        self.hyperparams = hyperparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")

        annotator_list = sorted(annotations_df['annotator_id'].unique())
        self.annotator_map = {name: i for i, name in enumerate(annotator_list)}
        self.num_annotators = len(self.annotator_map)

        annotations_df['annotator_idx'] = annotations_df['annotator_id'].map(self.annotator_map)

        # -1はアノテーションなしの意味
        pivoted_labels = annotations_df.pivot(index='item_id', columns='annotator_idx', values='label')
        self.noisy_labels_df = pivoted_labels.reindex(all_items).fillna(-1)
        self.features_df = features_df.reindex(all_items)

        self.model = LabelFusionNet(
            input_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            num_annotators=self.num_annotators,
            num_basis_matrices=self.hyperparams['M']
        ).to(self.device)

        # 学習率を0.0001に下げておくこと
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

        self.basis_matrices = self._generate_permutation_matrices(
            self.hyperparams['M'], NUM_CLASSES
        ).to(self.device)

    def _generate_permutation_matrices(self, M, K):
        matrices = torch.zeros(M, K, K)
        identity = torch.eye(K)
        for i in range(M):
            # 順列行列を生成
            matrices[i] = identity[torch.randperm(K)]
        return matrices

    def _compute_loss(self, logits, weights, coeffs, noisy_labels):
        P = torch.einsum('brm,mkl->brkl', coeffs, self.basis_matrices)

        # -1 (アノテーションなし) のラベルを無視するため、one-hot
        y_noisy_onehot = F.one_hot(noisy_labels.clamp(min=0), num_classes=NUM_CLASSES).float()

        # ノイズ行列 P を適用して、アノテータ r が真のラベル k を c と誤分類する確率のベクトルを得る
        y_clean = torch.matmul(P, y_noisy_onehot.unsqueeze(-1)).squeeze(-1)

        # アノテーションが存在する箇所のみマスク
        mask = (noisy_labels != -1).float().unsqueeze(-1)

        # アノテータの重みとマスクを適用してソフトターゲットを計算
        y_targ = torch.sum(weights.unsqueeze(-1) * y_clean * mask, dim=1)

        # ★★★ 数値安定化のための修正: ゼロ除算対策 ★★★
        y_targ = y_targ + EPSILON
        y_targ_sum = y_targ.sum(dim=1, keepdim=True)
        y_targ = y_targ / y_targ_sum # 正規化

        log_preds = F.log_softmax(logits, dim=1)

        # KLダイバージェンス損失
        # y_targにEPSILONを加えているため、Log(0)によるnan発生リスクが大幅に減少
        loss_kl = F.kl_div(log_preds, y_targ, reduction='batchmean')

        # 正則化項 (ノイズ行列 P の対角要素が1に近くなるように誘導させてみる)
        diag_P = torch.diagonal(P, dim1=-2, dim2=-1)
        reg_term = torch.sum((1.0 - diag_P)**2, dim=2)
        loss_reg = torch.mean(torch.sum(reg_term * mask.squeeze(-1), dim=1) / (mask.sum(dim=1).clamp(min=EPSILON)))

        return loss_kl + self.hyperparams['lambda'] * loss_reg

    def train(self):
        dataset = NoisyLabelsDataset(self.features_df, self.noisy_labels_df)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

        self.model.train()
        print("モデル学習開始...")
        for epoch in range(self.hyperparams['epochs']):
            total_loss = 0
            for features, noisy_labels in tqdm(dataloader, desc=f"エポック {epoch+1}/{self.hyperparams['epochs']}"):
                features, noisy_labels = features.to(self.device), noisy_labels.to(self.device)

                self.optimizer.zero_grad()
                logits, weights, coeffs = self.model(features)
                loss = self._compute_loss(logits, weights, coeffs, noisy_labels)

                # 損失がnanの場合、学習を中断
                if torch.isnan(loss):
                    print(f"\n エポック {epoch+1} で損失が nan の場合学習中断。")
                    return

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"エポック {epoch+1} | 平均損失: {total_loss / len(dataloader):.4f}")

    def predict(self):
        self.model.eval()
        dataset = NoisyLabelsDataset(self.features_df, self.noisy_labels_df)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for features, _ in tqdm(dataloader, desc="予測計算中"):
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
            print("\nアノテーションファイルのRead失敗。")
            return

        print("? 全ファイルの読み込み成功。")
        print(f" ?- アイテム総数: {len(all_items_list)}")
        print(f" ?- アノテーション総数: {len(annotations_df)}")
        print(f" ?- アノテーター数: {annotations_df['annotator_id'].nunique()}")
    except FileNotFoundError as e:
        print(f"エラー: {e}。ファイルが正しいディレクトリに存在するか。")
        return

    # ★★★ 学習率を 0.0001 に変更した。0.001では収束しない ★★★
    hyperparams = {
        'lr': 0.0001,
        'epochs': 20,
        'batch_size': 64,
        'lambda': 1.0,
        'M': 30
    }

    fusion_algorithm = SampleWiseLabelFusion(all_items_list, annotations_df, features_df, hyperparams)
    fusion_algorithm.train()

    # nanで学習が中断された場合は予測スキップ
    if not (hasattr(fusion_algorithm, 'model') and torch.isnan(fusion_algorithm.model.backbone[0].weight.data).any()):
        estimated_true_labels = fusion_algorithm.predict()

        print("\n--- ★ 推定された真のラベル (先頭20件) ---")
        print(estimated_true_labels.head(20))

        estimated_true_labels.to_csv('estimated_true_labels.csv', header=['estimated_label'], index_label='item_id')
        print("\n? 全データの推定ラベルが 'estimated_true_labels.csv' に保存済。")

        print("\n--- ★ 推定ラベルとアイテムプレフィックスの一致率 ---")
        item_prefixes = estimated_true_labels.index.str.replace(r'\d+', '', regex=True)
        matches = (item_prefixes == estimated_true_labels)
        total_items = len(estimated_true_labels)
        matched_count = matches.sum()

        print(f" ?- アイテム総数: {total_items}")
        print(f" ?- 一致したアイテム数: {matched_count}")
        if total_items > 0:
            print(f" ?- 一致率: {matched_count / total_items:.2%}")
        else:
            print(" ?- 一致率: 0.00%")

if __name__ == '__main__':
    main()