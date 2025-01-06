import matplotlib.pyplot as plt

# データを辞書に整形
data = {
    "5%": {
        "Fail": 0.002077306569719076,
        "Success": 0.0015578229984711668
    },
    "10%": {
        "Fail": 0.0018339481068613698,
        "Success": 0.001551228180141019
    },
    "15%": {
        "Fail": 0.0016053715712898116,
        "Success": 0.0015684455163047862
    },
     "20%": {
        "Fail":  0.0016364121122040664,
        "Success": 0.001641370056209287
    },
     "25%": {
        "Fail": 0.0017454754647671902,
        "Success": 0.001539001747086749
    },
     "30%": {
        "Fail":  0.0015516756467054385,
        "Success": 0.0015147340059823002
    }
}

# 横軸の値
x = ['5%', '10%', '15%', '20%', '25%', '30%']

# 縦軸の値（検出失敗と検出成功の平均値）

y_fail = [data[str(percent)]['Fail'] for percent in x]
y_success = [data[str(percent)]['Success'] for percent in x]

# 縦軸の範囲を指定 
plt.ylim(0.0010, 0.0023)

# グラフの描画
plt.plot(x, y_fail, label='Fail')
plt.plot(x, y_success, label='Success')

# グラフのタイトルと軸ラベルの設定
plt.title('Herlev:Comparison of PageRank(Initial label:70%)')
plt.xlabel('Percentage of label noise(%)')
plt.ylabel('PageRank Score')

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()



# データを辞書に整形
data = {
    "5%": {
        "Fail": 0.0018120750884045396,
        "Success": 0.0015562972817685986
    },
    "10%": {
        "Fail":  0.0016181454949774914,
        "Success": 0.0015241008055043644
    },
    "15%": {
        "Fail": 0.001598130939193607,
        "Success": 0.0014295999104452727
    },
     "20%": {
        "Fail":  0.0015605302031649563,
        "Success": 0.0015450983536325512
    },
     "25%": {
        "Fail": 0.0015721072072753825,
        "Success": 0.0014577116806025973
    },
     "30%": {
        "Fail":  0.0016030876041060135,
        "Success": 0.0015503065104296482
    }
}

# 横軸の値
x = ['5%', '10%', '15%', '20%', '25%', '30%']

# 縦軸の値（検出失敗と検出成功の平均値）

y_fail = [data[str(percent)]['Fail'] for percent in x]
y_success = [data[str(percent)]['Success'] for percent in x]

# 縦軸の範囲を指定 
plt.ylim(0.0010, 0.0023)

# グラフの描画
plt.plot(x, y_fail, label='Fail')
plt.plot(x, y_success, label='Success')

# グラフのタイトルと軸ラベルの設定
plt.title('Herlev:Comparison of PageRank(Initial label:50%)')
plt.xlabel('Percentage of label noise(%)')
plt.ylabel('PageRank Score')

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()


# データを辞書に整形
data = {
    "5%": {
        "Fail": 0.0010413644543261844,
        "Success": 0.0010067522341630005
    },
    "10%": {
        "Fail":  0.0011300304110230485,
        "Success": 0.0010838692452843444
    },
    "15%": {
        "Fail": 0.001112373825659717,
        "Success": 0.0009989780661333976
    },
     "20%": {
        "Fail":  0.0011355066929050373,
        "Success": 0.0010038589235802375
    },
     "25%": {
        "Fail": 0.0010158299987096087,
        "Success": 0.0010026698411601596
    },
     "30%": {
        "Fail":  0.001039236613567151,
        "Success": 0.0010295785849210144
    }
}

# 横軸の値
x = ['5%', '10%', '15%', '20%', '25%', '30%']

# 縦軸の値（検出失敗と検出成功の平均値）

y_fail = [data[str(percent)]['Fail'] for percent in x]
y_success = [data[str(percent)]['Success'] for percent in x]

# 縦軸の範囲を指定 
plt.ylim(0.0009, 0.0013)

# グラフの描画
plt.plot(x, y_fail, label='Fail')
plt.plot(x, y_success, label='Success')

# グラフのタイトルと軸ラベルの設定
plt.title('SIPaKMeD:Comparison of PageRank(Initial label:70%)')
plt.xlabel('Percentage of label noise(%)')
plt.ylabel('PageRank Score')

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()



# データを辞書に整形
data = {
    "5%": {
        "Fail": 0.001137305381916031,
        "Success": 0.0010962934225352717
    },
    "10%": {
        "Fail":  0.0009604590856653591,
        "Success": 0.000962453192180516
    },
    "15%": {
        "Fail": 0.0010453692770705382,
        "Success": 0.001018640585328159
    },
     "20%": {
        "Fail":  0.0011701426360855162,
        "Success": 0.0010587284100629939
    },
     "25%": {
        "Fail": 0.0010580849731293021,
        "Success": 0.001032961469536432
    },
     "30%": {
        "Fail":  0.001034412305955316,
        "Success": 0.0010314668758485586
    }
}

# 横軸の値
x = ['5%', '10%', '15%', '20%', '25%', '30%']

# 縦軸の値（検出失敗と検出成功の平均値）

y_fail = [data[str(percent)]['Fail'] for percent in x]
y_success = [data[str(percent)]['Success'] for percent in x]

# 縦軸の範囲を指定 
plt.ylim(0.0009, 0.0013)

# グラフの描画
plt.plot(x, y_fail, label='Fail')
plt.plot(x, y_success, label='Success')

# グラフのタイトルと軸ラベルの設定
plt.title('SIPaKMeD:Comparison of PageRank(Initial label:50%)')
plt.xlabel('Percentage of label noise(%)')
plt.ylabel('PageRank Score')

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()