import os

def rename_png_files(folder_path):

#�ŏ��Ɋ�����png�t�@�C�����폜����
# for filename in os.listdir("."):
#    if filename.endswith('.png'):
#      os.remove(os.path.join(folder_path, filename))
#      print(f"�폜���܂���: {filename}")

  # �t�H���_���̃t�@�C���ꗗ��\��
  #print(f"���݂̃t�H���_ {folder_path} ����png�t�@�C��:")
 png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
  # for file in png_files:
  #  print(file)

  # png�t�@�C������ύX
 count = 1
 for filename in png_files:
        new_name = f"{folder_path}/C-NI-{count:03}.png"
        os.rename(os.path.join(folder_path, filename), new_name)
        count += 1


# �t�H���_�p�X���w��

folder_path = "."

rename_png_files(folder_path)