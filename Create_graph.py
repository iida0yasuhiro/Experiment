import matplotlib.pyplot as plt
import numpy as np
from builtins import min

'''
def calculate_differences(file_path):
    """
    ファイルから数値を読み込み、上から2行ずつ差分を計算し、
    絶対値と値の小さい方を一緒にリストに格納する。

    Args:
        file_path: 読み込むファイルのパス

    Returns:
        tuple: 計算結果のタプルのリストと最小値のリストを含むタプル
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        numbers = [float(line.strip().split()[1]) for line in lines]

        results = []
        mins = []
        for i in range(0, len(numbers), 2):
            num1, num2 = numbers[i], numbers[i+1]
            diff = abs(num2 - num1)
            min_value = min(num1, num2)
            results.append(diff)
            mins.append(min_value)

    return results, mins
'''

def calculate_differences(file_path):
    """
    Precision: で始まる行の数値を読み込み、上から2行ずつ差分を計算し、
    絶対値と値の小さい方を一緒にリストに格納。

    Args:
        file_path: 読み込むファイルのパス

    Returns:
        tuple: 計算結果のタプルのリストと最小値のリストを含むタプル
    """

    results = []
    mins = []
    with open(file_path, 'r') as f:
        numbers = []
        for line in f:
            if line.startswith("Precision: "):
                try:
                    # "Precision: " の後の数値を抽出
                    number = float(line.split(": ")[1])
                    numbers.append(number)
                except ValueError:
                    print(f"数値に変換できませんでした: {line}")

            # 2つ以上の数値が蓄積されたら計算
            if len(numbers) >= 2:
                num1, num2 = numbers.pop(0), numbers.pop(0)
                diff = abs(num2 - num1)
                min_value = min(num1, num2)
                results.append(diff)
                mins.append(min_value)

    return results, mins


# ファイルパスを指定
file_path = "Precision-4049.txt"

# 差分を計算
result, min = calculate_differences(file_path)

print(result)
print(len(result))

def split_list_every_three(data):
  """
  与えられたリストを3つ飛ばしで3つのリストに分割。
  Args:
    data: 分割するリスト
  Returns:
    tuple: 3つのリストを含むタプル (list_a, list_b, list_c)
  """

  if len(data) < 9:
    raise ValueError("リストの長さは少なくとも9以上である必要")

  list_a = data[::3]
  list_b = data[1::3]
  list_c = data[2::3]

  return list_a, list_b, list_c

def extract_precision_values(filename):
  """
  Extracts precision values from a file containing lines like "Precision: 0.95".

  Args:
    filename: The path to the file containing the precision values.

  Returns:
    A list of precision values as floats.
  """
  precision_values = []
  with open(filename, 'r') as file:
    for line in file:
      if line.startswith("Precision: "):
        precision = float(line.split(": ")[1])
        precision_values.append(precision)
  return precision_values

# リストを分割
yerr_10_90, yerr_20_80, yerr_30_70 = split_list_every_three(result)

print("yerr_10_90:", yerr_10_90)
print("yerr_20_80:", yerr_20_80)
print("yerr_30_70:", yerr_30_70)

center_a, center_b, center_c = split_list_every_three(min)

x = ['5%', '10%', '15%', '20%', '25%', '30%']

a = center_a
#+ np.array(yerr_10_90) / 2
b = center_b
#+ np.array(yerr_20_80) / 2
c = center_c
#+ np.array(yerr_30_70) / 2

# グラフを描画
plt.errorbar(x, a, yerr=np.array(yerr_10_90)/2, fmt='-', capsize=5,label='Initial label ratio 68.9%(90% of Abnormal / 10% of Normal')
plt.errorbar(x, b, yerr=np.array(yerr_20_80)/2, fmt='-', capsize=5,label='Initial label ratio 64.2%(80% of Abnormal / 20% of Normal')
plt.errorbar(x, c, yerr=np.array(yerr_30_70)/2, fmt='-', capsize=5,label='Initial label ratio 59.4%(70% of Abnormal / 30% of Normal',color=(0.5, 0.5, 0.5))

# グラフのタイトルと軸ラベル
plt.title('Precision with different label noise ratio (Herlev)')
plt.xlabel('Label noise ratio')
plt.ylabel('Precision')

# 凡例を表示
plt.legend()
plt.ylim(0.0, 1.01)

# グリッドを表示
plt.grid(True)

# グラフを表示
plt.show()


