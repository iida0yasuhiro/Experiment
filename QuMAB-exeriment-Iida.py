# 7月16日 QuMABの細胞診用のQ-Former（論文をもとに実装）

import os
import json
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance, entropy


# データを読み込む
def prepare_data(feat_json, raw_noise_json, class_to_idx):
    """
    画像を標準化した特徴量、アノテーターID、画像ID、アノテーターの決定。
    """
    if not os.path.exists(feat_json) or not os.path.exists(raw_noise_json):
        raise FileNotFoundError(f"データファイルが見つかりません。{feat_json} と {raw_noise_json} が存在することを確認してください。")

    with open(feat_json, 'r') as f:
        features = json.load(f)
    with open(raw_noise_json, 'r') as f:
        raw_data = ast.literal_eval(f.read())

    X = StandardScaler().fit_transform(np.array(list(features.values()))).astype(np.float32)
    name_to_idx = {os.path.splitext(os.path.basename(n))[0]: i for i, n in enumerate(features.keys())}

    ann_ids, decisions, image_ids = [], [], []
    gt = np.zeros(len(X), dtype=int)

    for k, p_dict in enumerate(raw_data):
        for c_name, sub_dict in p_dict.items():
            for i_name, label in sub_dict.items():
                c_name_clean = os.path.splitext(os.path.basename(i_name))[0]
                if c_name_clean in name_to_idx:
                    img_idx = name_to_idx[c_name_clean]
                    ann_ids.append(k)
                    image_ids.append(img_idx)
                    decisions.append(int(label))
                    gt[img_idx] = class_to_idx[c_name]

    return X, np.array(ann_ids), np.array(image_ids), np.array(decisions), gt


#QuMABモデル定義（Cross-AttentionとQ-Formerを、ここで実装した

class QFormerLayer(nn.Module):
    """
    Q-Formerの1層分：Cross-Attention（マルチヘッド） -> FFN
    論文のInstructBLIP由来Q-Formerにおける
    「クエリが画像特徴と相互作用し、非線形変換を経る」1層分の処理に対応。
    """
    def __init__(self, chunk_dim, feature_dim, n_heads_cross, dropout=0.2):
        super().__init__()
        # [選択肢A] マルチヘッドCross-Attention（論文 "typically with 12 heads" に対応）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=chunk_dim, num_heads=n_heads_cross, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(chunk_dim)

        # FFN（Q-Former構造に倣い、Cross-Attention出力を非線形変換）
        self.ffn = nn.Sequential(
            nn.Linear(chunk_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, chunk_dim),
        )
        self.norm_ffn = nn.LayerNorm(chunk_dim)

    def forward(self, query, pseudo_tokens):
        """
        query:         (batch, 1, chunk_dim)
        pseudo_tokens:  (batch, n_chunks, chunk_dim)
        """
        attn_out, attn_weights = self.cross_attn(query, pseudo_tokens, pseudo_tokens)
        query = self.norm_attn(query + attn_out)

        ffn_out = self.ffn(query)
        query = self.norm_ffn(query + ffn_out)

        return query, attn_weights  # (batch, 1, chunk_dim), (batch, 1, n_chunks)


class QuMABModel(nn.Module):
    """
    アーキテクチャ：
      annotator_queries -> Self-Attention（アノテータ間相関）
                         -> [Cross-Attention(マルチヘッド) -> FFN] × N_LAYERS層
                         -> Annotator別Classifier

    n_heads_cross=12（論文仕様）。chunk_dimがn_heads_crossで割り切れるよう
              n_chunks=8（→chunk_dim=768/8=96, 96/12=8）をデフォルトに変更。
    n_layers（デフォルト4）でQ-Former風の多層構造を再現。
              各層でCross-Attention→FFNを繰り返し、クエリ表現を段階的に精緻化する。
    """
    def __init__(self, num_annotators, feature_dim, num_classes,
                 n_heads_self=4, n_chunks=8, n_heads_cross=12, n_layers=12, dropout=0.2):
        super().__init__()
        assert feature_dim % n_chunks == 0, \
            f"feature_dim({feature_dim})はn_chunks({n_chunks})で割り切れないといけない"
        chunk_dim = feature_dim // n_chunks
        assert chunk_dim % n_heads_cross == 0, \
            f"chunk_dim({chunk_dim})はn_heads_cross({n_heads_cross})で割り切ないといけない"

        self.num_annotators = num_annotators
        self.feature_dim = feature_dim
        self.n_chunks = n_chunks
        self.chunk_dim = chunk_dim
        self.n_layers = n_layers

        # アノテータ固有のクエリ（学習可能埋め込み）
        self.annotator_queries = nn.Embedding(num_annotators, feature_dim)

        # Self-Attention：アノテータ間相関（ヘッド数は feature_dim=768 用）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=n_heads_self, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # Query（feature_dim）をトークン空間（chunk_dim）に射影
        self.query_proj = nn.Linear(feature_dim, chunk_dim)

        # Q-Former層をN_LAYERS個積み重ねてみる　12が最適か
        #    各層：マルチヘッドCross-Attention→ FFN
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(chunk_dim, feature_dim, n_heads_cross, dropout)
            for _ in range(n_layers)
        ])

        # トークン空間 -> 元次元へ戻す最終射影
        self.output_proj = nn.Linear(chunk_dim, feature_dim)
        self.norm_out = nn.LayerNorm(feature_dim)

        # Annotator別に独立したClassifier
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            ) for _ in range(num_annotators)
        ])

    def get_correlated_queries(self):
        """Self-Attention適用後の、アノテータ間相関を反映したクエリを返す"""
        all_q = self.annotator_queries.weight.unsqueeze(0)      # (1, num_ann, feat_dim)
        attn_out, _ = self.self_attn(all_q, all_q, all_q)
        correlated = self.norm1(all_q + attn_out).squeeze(0)    # (num_ann, feat_dim)
        return correlated

    def forward(self, ann_ids, features, return_attn=False):
        """
        ann_ids:  (batch,)             アノテータID
        features: (batch, feature_dim) 画像特徴（768次元特徴）
        """
        batch_size = features.size(0)

        # Self-Attention適用後のクエリを取得 
        correlated_q = self.get_correlated_queries()            # (num_ann, feat_dim)
        batch_queries = correlated_q[ann_ids]                   # (batch, feat_dim)

        # トークン化 
        query = self.query_proj(batch_queries).unsqueeze(1)     # (batch, 1, chunk_dim)
        pseudo_tokens = features.view(batch_size, self.n_chunks, self.chunk_dim)
        # (batch, n_chunks, chunk_dim) ← Key/Value（本物の複数トークン）

        # ここでN_LAYERS層のQ-Formerを通すことにした
        last_attn_weights = None
        for layer in self.qformer_layers:
            query, attn_weights = layer(query, pseudo_tokens)
            last_attn_weights = attn_weights  # (batch, 1, n_chunks)（マルチヘッド平均済み）

        # 元次元に戻しておく
        annotator_features = self.norm_out(self.output_proj(query.squeeze(1)))  # (batch, feat_dim)

        # 最後にAnnotator別の分類
        num_classes = self.classifiers[0][-1].out_features
        out = torch.zeros(batch_size, num_classes, device=features.device)
        for i in range(self.num_annotators):
            mask = (ann_ids == i)
            if mask.any():
                out[mask] = self.classifiers[i](annotator_features[mask])

        if return_attn:
            return out, last_attn_weights.squeeze(1)  # (batch, n_chunks) 最終層の各チャンクへの注目度
        return out


# 混同行列作成、EMDおよびJSDで評価

def compute_normalized_cm(gt_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes))
    for g, p in zip(gt_labels, pred_labels):
        cm[g, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums + 1e-9, out=np.zeros_like(cm), where=row_sums != 0)
    return cm_norm


def compute_cm_distributions_metrics(cm_true, cm_pred, num_classes):
    """
    Bethesda分類の特性に合わせ、真の混同行列とモデル予測の混同行列を比較。
    行ごとに「EMD（幾何学的距離）」と「JSD（情報理論的確率発散）」を算出し、その平均値を計算。
    """
    row_emds = []
    row_jsds = []
    class_values = np.arange(num_classes)

    for r in range(num_classes):
        p_true = cm_true[r]
        p_pred = cm_pred[r]

        if p_true.sum() == 0 or p_pred.sum() == 0:
            continue

        emd_val = wasserstein_distance(class_values, class_values, p_true, p_pred)
        row_emds.append(emd_val)

        m = 0.5 * (p_true + p_pred)
        jsd_val = 0.5 * entropy(p_true, m, base=2) + 0.5 * entropy(p_pred, m, base=2)
        row_jsds.append(jsd_val)

    mean_emd = np.mean(row_emds) if len(row_emds) > 0 else np.nan
    mean_jsd = np.mean(row_jsds) if len(row_jsds) > 0 else np.nan
    return mean_emd, mean_jsd


#きちんとホールドアウトで検証しておく + EMD/JSD評価

if __name__ == "__main__":
    class_to_idx = {'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'ASC-H': 3, 'HSIL': 4, 'SCC': 5}
    num_classes = len(class_to_idx)

    print("--- データの読み込みと前処理中 ---")
    try:
        X, ann_ids, image_ids, decisions, gt = prepare_data(
            'SM-official-updated.json',
            'raw_noise_patterns-all-1.json',
            class_to_idx
        )
    except FileNotFoundError as e:
        print(e)
        exit(1)

    num_annotators = len(np.unique(ann_ids))
    feature_dim = X.shape[1]
    print(f"総サンプル数（アノテーション数）: {len(decisions)}")
    print(f"ユニーク画像数: {len(X)}")
    print(f"アノテーター数: {num_annotators}")
    print(f"特徴量次元: {feature_dim}")

    # データの分割（リークがないことを検証する） 
    #np.random.seed(42)
    #torch.manual_seed(42)

    num_samples = len(decisions)
    shuffled_indices = np.random.permutation(num_samples)

    train_size = int(num_samples * 0.8)
    train_idx = shuffled_indices[:train_size]
    test_idx = shuffled_indices[train_size:]

    X_t = torch.FloatTensor(X)
    ann_t = torch.LongTensor(ann_ids)
    dec_t = torch.LongTensor(decisions)
    image_ids_t = torch.LongTensor(image_ids)

    #  n_chunks=8 → chunk_dim=96（=768/8）は12ヘッドで割り切れる（96/12=8）
    # n_layers → Q-Formerを12層で実験

    N_CHUNKS = 8
    N_HEADS_CROSS = 12
    N_LAYERS = 12

    model = QuMABModel(num_annotators, feature_dim, num_classes,
                        n_chunks=N_CHUNKS, n_heads_cross=N_HEADS_CROSS, n_layers=N_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 学習開始
    print(f"\n--- QuMABモデル(Self-Attn + [Cross-Attn({N_HEADS_CROSS}heads)->FFN]×{N_LAYERS}層 + "
          f"Annotator別Classifier)学習中 (Trainデータのみ使用) ---")
    num_epochs = 300
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(ann_t[train_idx], X_t[image_ids_t[train_idx]])
        loss = criterion(outputs, dec_t[train_idx])

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}] - Train Loss: {loss.item():.4f}")

    # 評価
    model.eval()
    results = []
    all_chunk_attn = {}  # アノテータごとのチャンク注目度（可視化・分析用）

    print("\n--- テストデータに対するアノテーター予測・混同行列評価中 (EMD / JSD) ---")
    with torch.no_grad():
        for i in range(num_annotators):
            test_ann_indices = test_idx[ann_ids[test_idx] == i]

            if len(test_ann_indices) == 0:
                print(f"User_{i}: テストデータが存在しないためスキップ。")
                continue

            y_true = decisions[test_ann_indices]
            preds_logits, chunk_attn = model(
                ann_t[test_ann_indices], X_t[image_ids_t[test_ann_indices]],
                return_attn=True
            )
            preds = preds_logits.argmax(dim=1).numpy()

            # アノテータiの平均チャンク注目度を記録（Figure用）
            all_chunk_attn[i] = chunk_attn.mean(dim=0).numpy()

            y_gt = gt[image_ids[test_ann_indices]]

            cm_true = compute_normalized_cm(y_gt, y_true, num_classes)
            cm_pred = compute_normalized_cm(y_gt, preds, num_classes)

            emd_score, jsd_score = compute_cm_distributions_metrics(cm_true, cm_pred, num_classes)

            results.append({
                'Annotator': f'User_{i}',
                'Test_Samples': len(test_ann_indices),
                'CM_EMD': emd_score,
                'CM_JSD': jsd_score
            })

    df = pd.DataFrame(results)

    print("\n" + "=" * 88)
    print("       【QuMAB (マルチヘッドCross-Attention + 多層Q-Former統合)：")
    print("        人間の認知ノイズ分布近似度評価結果（未知データ）】")
    print("   CM_EMD（幾何学的距離）/ CM_JSD（情報理論的確率発散）ともに低いほどよい")
    print(f"   アノテータ間相関 + [Cross-Attn({N_HEADS_CROSS}heads, n_chunks={N_CHUNKS})")
    print(f"      -> FFN] × {N_LAYERS}層 + Annotator別Classifierを実装")
    print("   空間的な位置情報（画像のどこを見たか）は再現できないが、")
    print("      Cross-Attention機構自体（マルチヘッドの複数トークンへの選択的重み付け）と")
    print("      Q-Formerの多層構造を再現した")
    print("=" * 88)
    print(df.to_string(index=False, na_rep='N/A', float_format=lambda x: f"{x:.3f}"))
    print("-" * 88)

    mean_emd = df['CM_EMD'].dropna().mean()
    mean_jsd = df['CM_JSD'].dropna().mean()
    print(f"【全体平均】 CM_EMD: {mean_emd:.3f} | CM_JSD: {mean_jsd:.3f}")
    print("=" * 88 + "\n")

    # チャンク注目度の確認
    print("【アノテータ別：チャンク注目度（検証）】")
    print(f"  一様分布の理論値: {1/N_CHUNKS:.4f}（全チャンクが均等なら退化＝無意味な注意）")
    for i, attn in all_chunk_attn.items():
        print(f"  User_{i}: 分散={attn.var():.6f}  最大={attn.max():.3f}  最小={attn.min():.3f}")

    np.save('qumab_chunk_attention.npy', all_chunk_attn)
    print("\n[完了] チャンク注目度データを qumab_chunk_attention.npy に保存完了。")