import matplotlib.pyplot as plt
import numpy as np
from builtins import min

'''
def calculate_differences(file_path):
    """
    �t�@�C�����琔�l��ǂݍ��݁A�ォ��2�s���������v�Z���A
    ��Βl�ƒl�̏����������ꏏ�Ƀ��X�g�Ɋi�[���܂��B

    Args:
        file_path: �ǂݍ��ރt�@�C���̃p�X

    Returns:
        tuple: �v�Z���ʂ̃^�v���̃��X�g�ƍŏ��l�̃��X�g���܂ރ^�v��
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
    Precision: �Ŏn�܂�s�̐��l��ǂݍ��݁A�ォ��2�s���������v�Z���A
    ��Βl�ƒl�̏����������ꏏ�Ƀ��X�g�Ɋi�[���܂��B

    Args:
        file_path: �ǂݍ��ރt�@�C���̃p�X

    Returns:
        tuple: �v�Z���ʂ̃^�v���̃��X�g�ƍŏ��l�̃��X�g���܂ރ^�v��
    """

    results = []
    mins = []
    with open(file_path, 'r') as f:
        numbers = []
        for line in f:
            if line.startswith("Precision: "):
                try:
                    # "Precision: " �̌�̐��l�𒊏o
                    number = float(line.split(": ")[1])
                    numbers.append(number)
                except ValueError:
                    print(f"���l�ɕϊ��ł��܂���ł���: {line}")

            # 2�ȏ�̐��l���~�ς��ꂽ��v�Z
            if len(numbers) >= 2:
                num1, num2 = numbers.pop(0), numbers.pop(0)
                diff = abs(num2 - num1)
                min_value = min(num1, num2)
                results.append(diff)
                mins.append(min_value)

    return results, mins


# �t�@�C���p�X���w��
file_path = "Precision-4049.txt"

# �������v�Z
result, min = calculate_differences(file_path)

print(result)
print(len(result))

def split_list_every_three(data):
  """
  �^����ꂽ���X�g��3��΂���3�̃��X�g�ɕ����B
  Args:
    data: �������郊�X�g
  Returns:
    tuple: 3�̃��X�g���܂ރ^�v�� (list_a, list_b, list_c)
  """

  if len(data) < 9:
    raise ValueError("���X�g�̒����͏��Ȃ��Ƃ�9�ȏ�ł���K�v")

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

# ���X�g�𕪊�
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

# �O���t��`��
plt.errorbar(x, a, yerr=np.array(yerr_10_90)/2, fmt='-', capsize=5,label='Initial label ratio 68.9%(90% of Abnormal / 10% of Normal')
plt.errorbar(x, b, yerr=np.array(yerr_20_80)/2, fmt='-', capsize=5,label='Initial label ratio 64.2%(80% of Abnormal / 20% of Normal')
plt.errorbar(x, c, yerr=np.array(yerr_30_70)/2, fmt='-', capsize=5,label='Initial label ratio 59.4%(70% of Abnormal / 30% of Normal',color=(0.5, 0.5, 0.5))

# �O���t�̃^�C�g���Ǝ����x��
plt.title('Precision with different label noise ratio (Herlev)')
plt.xlabel('Label noise ratio')
plt.ylabel('Precision')

# �}���\��
plt.legend()
plt.ylim(0.0, 1.01)

# �O���b�h��\��
plt.grid(True)

# �O���t��\��
plt.show()

