import numpy as np

para = 3
fout = open("./politihop_scripts/run_para_{}.sh".format(para), 'w')
print("export CUDA_VISIBLE_DEVICES={}".format(para+1), file=fout)

for constraint in np.arange(0, 0.01, 0.001):
    for claim in np.arange(0, 1, 0.1):
        for scaled in [0.8, 0.9, 1.0, 1.1, 1.2]:
            print("python politihop_scripts/dual_debiased_model.py --constraint_loss_weight {} --claim_loss_weight {} --scaled_rate {}".format(round(constraint, 3), round(claim, 3), scaled), file=fout)