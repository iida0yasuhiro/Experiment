import torch
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import numpy as np
import os
import json

# Herlevデータ用（7分類）

# 最初に既存のファイルを削除する
#
folder_path = "."

#for filename in os.listdir("."):
#     if filename.endswith('.bmp'):
#       os.remove(os.path.join(folder_path, filename))
#       print(f"削除しました: {filename}")



# モデルのロード
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to (224, 224) to match model input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 画像ファイルのパス
#image_paths = ["./A-001.BMP","./A-002.BMP","./A-003.BMP"]

image_paths = []

num_files = 523

i=1

# 読み込むファイル名を指定しておく
for i in range(1, num_files):
    file_name = f"./C-HS-{i:03}.png"  # 3桁でゼロ埋め
    image_paths.append(file_name)

#print(image_paths)


node_dict = {}

# 各画像の特徴ベクトルを抽出
for image_path in image_paths:
    image = Image.open(image_path)
    input_ids = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state  # 最終層の出力

    # クラストークンの特徴ベクトルを抽出 (通常は分類タスクで利用)
    feature_vector = last_hidden_states[:, 0, :].numpy()[0]

    # 特徴ベクトルの各要素をリストで表示
    node_dict[image_path] = feature_vector.tolist()

    #num = len(feature_vector.tolist())
    #print(num)
    #print(f"{image_path},{feature_vector}")
    #print(feature_vector.tolist())
    #print(node_dict)

    # JSONファイルに書き出し
    with open('C-HS.json', 'w') as f:
      json.dump(node_dict, f, indent=4)

