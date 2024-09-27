import numpy as np

para = 3
fout = open("./politihop_scripts/run_politihop_3.sh".format(para), 'w')
print("export CUDA_VISIBLE_DEVICES={}".format(para+1), file=fout)

for constraint in np.arange(0, 0.01, 0.001):
    for claim in np.arange(0, 1, 0.1):
        print("python politihop_scripts/dual_debiased_model_3.py --constraint_loss_weight {} --claim_loss_weight {}".format(round(constraint, 3), round(claim, 3)), file=fout)