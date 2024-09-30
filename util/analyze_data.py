import argparse
import pandas as pd
import matplotlib.pyplot as plt

def analyze_constraint_claim(in_path, out_path):
    # 读取文本文件
    with open(in_path, 'r') as file:
        lines = file.readlines()

    # 创建空的数据列表
    data = []

    # 提取数据
    for line in lines:
        line = line.strip()
        if line.startswith('constraint_loss_weight:'):
            con_lw, claim_lw, _ = line.split(',')
            constraint_loss_weight = float(con_lw.strip().split(':')[1].strip())
            claim_loss_weight = float(claim_lw.strip().split(':')[1].strip())
        elif line.startswith('Accuracy:'):
            accuracy = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('F1 (micro):'):
            f1_micro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Precision (macro):'):
            precision_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Recall (macro):'):
            recall_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('F1 (macro):'):
            f1_macro = float(line.split(':')[1].strip().replace('%', ''))
            data.append([constraint_loss_weight, claim_loss_weight, accuracy, f1_micro, precision_macro, recall_macro, f1_macro])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['constraint_loss_weight', 'claim_loss_weight', 'Accuracy', 'F1 (micro)', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)'])
    df_sorted = df.sort_values(by=['constraint_loss_weight', 'claim_loss_weight'], ascending=[True, True])
    df_rounded = df_sorted.round(decimals=3)
    
    df_rounded.to_excel(out_path)

    # 打印DataFrame
    print(df_rounded)

def analyze_dual_unbias_result(in_path, out_path):
    # 读取文本文件
    with open(in_path, 'r') as file:
        lines = file.readlines()

    # 创建空的数据列表
    data = []

    # 提取数据
    for line in lines:
        line = line.strip()
        if line.startswith('constraint_claim_loss_weight:'):
            cc_lw, ce_lw, clw, elw = line.split(',')
            constraint_claim_loss_weight = float(cc_lw.strip().split(':')[1].strip())
            constraint_evidence_loss_weight = float(ce_lw.strip().split(':')[1].strip())
            claim_loss_weight = float(clw.strip().split(':')[1].strip())
            evidence_loss_weight = float(elw.strip().split(':')[1].strip())
        elif line.startswith('Accuracy:'):
            accuracy = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('F1 (micro):'):
            f1_micro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Precision (macro):'):
            precision_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('Recall (macro):'):
            recall_macro = float(line.split(':')[1].strip().replace('%', ''))
        elif line.startswith('F1 (macro):'):
            f1_macro = float(line.split(':')[1].strip().replace('%', ''))
            data.append([constraint_claim_loss_weight, constraint_evidence_loss_weight, claim_loss_weight, evidence_loss_weight, accuracy, f1_micro, precision_macro, recall_macro, f1_macro])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['constraint_claim_loss_weight', 'constraint_evidence_loss_weight', 'claim_loss_weight', 'evidence_loss_weight', 'Accuracy', 'F1 (micro)', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)'])
    df_sorted = df.sort_values(by=['constraint_claim_loss_weight', 'constraint_evidence_loss_weight', 'claim_loss_weight', 'evidence_loss_weight'], ascending=[True, True, True, True])
    df_rounded = df_sorted.round(decimals=3)
    
    df_rounded.to_excel(out_path)

    # 打印DataFrame
    print(df_rounded)

def main(args):
    # draw_constraint_claim(args.in_path, args.out_path, args.figure_path)
    # analyze_dual_unbias_result(args.in_path, args.out_path)
    analyze_constraint_claim(args.in_path, args.out_path)
    # draw_constraint(args.figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for unbias
    # parser.add_argument("--in_path", type=str, default='/data/yangjun/fact/debias/para_results/self_supervised_unbiased_model.txt')
    # parser.add_argument("--out_path", type=str, default='/data/yangjun/fact/debias/results/two_CHEF_self_supervised_unbiased.xlsx')
    # parser.add_argument("--figure_path", type=str, default='/data/yangjun/fact/debias/results/two_CHEF_self_supervised_unbiased.png')

    # for politihop_3
    # parser.add_argument("--in_path", type=str, default='para_results/politihop_3.txt')
    # parser.add_argument("--out_path", type=str, default='./save_logs/results/politihop_3.xlsx')

    # for adversarial_politihop_2
    # parser.add_argument("--in_path", type=str, default='para_results/adversarial_politihop_2.txt')
    # parser.add_argument("--out_path", type=str, default='./save_logs/results/adversarial_politihop_2.xlsx')

    # for symmetric_politihop_2
    # parser.add_argument("--in_path", type=str, default='para_results/symmetric_politihop.txt')
    # parser.add_argument("--out_path", type=str, default='./save_logs/results/symmetric_politihop.xlsx')

    # for symmetric_politihop_shareevi_2
    parser.add_argument("--in_path", type=str, default='para_results/symmetric_politihop_shareevi.txt')
    parser.add_argument("--out_path", type=str, default='./save_logs/results/symmetric_politihop_shareevi.xlsx')

    args = parser.parse_args()
    main(args)
