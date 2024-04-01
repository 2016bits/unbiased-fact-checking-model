import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_constraint_claim(in_path, out_path, figure_path):
    # 读取文本文件
    with open(in_path, 'r') as file:
        lines = file.readlines()

    # 创建空的数据列表
    data = []

    # 提取数据
    for line in lines:
        line = line.strip()
        if line.startswith('constraint_loss_weight:'):
            con_lw, claim_lw = line.split(',')
            constraint_loss_weight = float(con_lw.strip().split(':')[1].strip())
            claim_loss_weight = float(claim_lw.strip().split(':')[1].strip())
        elif line.startswith('F1 (micro):'):
            f1_micro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Precision (macro):'):
            precision_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Recall (macro):'):
            recall_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('F1 (macro):'):
            f1_macro = float(line.split(':')[1].strip().replace('%', ''))
            data.append([constraint_loss_weight, claim_loss_weight, f1_micro, precision_macro, recall_macro, f1_macro])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['constraint_loss_weight', 'claim_loss_weight', 'F1 (micro)', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)'])
    df_sorted = df.sort_values(by=['constraint_loss_weight', 'claim_loss_weight'], ascending=[True, True])
    df_rounded = df_sorted.round(decimals=3)
    
    df_rounded.to_excel(out_path)

    # 打印DataFrame
    print(df_rounded)

def draw_constraint(figure_path):
    constraint_loss_weight = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    f1_micro = [80.962, 81.263, 80.661, 80.16, 80.261, 80.962, 80.962, 81.463, 80.762, 80.361]
    precision_macro = [83.187, 83.344, 82.94, 82.391, 82.749, 83.203, 82.928, 83.862, 82.874, 82.517]
    recall_macro = [80.943, 81.242, 80.64, 80.14, 80.24, 80.941, 80.941, 81.442, 80.741, 80.34]
    f1_macro = [80.623, 80.781, 80.107, 79.703, 79.861, 80.44, 80.469, 81.003, 80.276, 79.887]

    plt.plot(constraint_loss_weight, f1_micro, label='F1 (micro)')
    plt.plot(constraint_loss_weight, precision_macro, label='Precision (macro)')
    plt.plot(constraint_loss_weight, recall_macro, label='Recall (macro)')
    plt.plot(constraint_loss_weight, f1_macro, label='F1 (macro)')

    plt.xlabel('constraint_loss_weight')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    plt.savefig(figure_path)

def main(args):
    # draw_constraint_claim(args.in_path, args.out_path, args.figure_path)
    draw_constraint(args.figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/data/yangjun/fact/debias/logs/test_result.txt')
    parser.add_argument("--out_path", type=str, default='/data/yangjun/fact/debias/results/parameter_results.xlsx')
    parser.add_argument("--figure_path", type=str, default='/data/yangjun/fact/debias/results/parameter_results.png')

    args = parser.parse_args()
    main(args)
