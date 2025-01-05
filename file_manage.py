import os

def rename_png_files(folder_path):

#最初に既存のpngファイルを削除する
# for filename in os.listdir("."):
#    if filename.endswith('.png'):
#      os.remove(os.path.join(folder_path, filename))
#      print(f"削除しました: {filename}")

  # フォルダ内のファイル一覧を表示
  #print(f"現在のフォルダ {folder_path} 内のpngファイル:")
 png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
  # for file in png_files:
  #  print(file)

  # pngファイル名を変更
 count = 1
 for filename in png_files:
        new_name = f"{folder_path}/C-NI-{count:03}.png"
        os.rename(os.path.join(folder_path, filename), new_name)
        count += 1


# フォルダパスを指定

folder_path = "."

rename_png_files(folder_path)