import torch
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import numpy as np
import os
import json

# Herlev�f�[�^�p�i7���ށj

# �ŏ��Ɋ����̃t�@�C�����폜����
#
folder_path = "."

#for filename in os.listdir("."):
#     if filename.endswith('.bmp'):
#       os.remove(os.path.join(folder_path, filename))
#       print(f"�폜���܂���: {filename}")



# ���f���̃��[�h
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# �O����
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to (224, 224) to match model input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# �摜�t�@�C���̃p�X
#image_paths = ["./A-001.BMP","./A-002.BMP","./A-003.BMP"]

image_paths = []

num_files = 523

i=1

# �ǂݍ��ރt�@�C�������w�肵�Ă���
for i in range(1, num_files):
    file_name = f"./C-HS-{i:03}.png"  # 3���Ń[������
    image_paths.append(file_name)

#print(image_paths)


node_dict = {}

# �e�摜�̓����x�N�g���𒊏o
for image_path in image_paths:
    image = Image.open(image_path)
    input_ids = transform(image).unsqueeze(0)  # �o�b�`������ǉ�

    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state  # �ŏI�w�̏o��

    # �N���X�g�[�N���̓����x�N�g���𒊏o (�ʏ�͕��ރ^�X�N�ŗ��p)
    feature_vector = last_hidden_states[:, 0, :].numpy()[0]

    # �����x�N�g���̊e�v�f�����X�g�ŕ\��
    node_dict[image_path] = feature_vector.tolist()

    #num = len(feature_vector.tolist())
    #print(num)
    #print(f"{image_path},{feature_vector}")
    #print(feature_vector.tolist())
    #print(node_dict)

    # JSON�t�@�C���ɏ����o��
    with open('C-HS.json', 'w') as f:
      json.dump(node_dict, f, indent=4)

