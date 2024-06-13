import numpy as np

para = 5
fout = open("./run_scripts/dual_unbias/run_para_{}.sh".format(para), 'w')
print("export CUDA_VISIBLE_DEVICES={}".format(para+1), file=fout)

for cc in [0.001,0.0001]:
    for ce in [0.1,0.01,0.001,0.0001]:
        for claim in [0.1,0.01,0.001]:
            for evidence in [0.1,0.01,0.001]:
                print("python two_class_scripts/dual_unbiased_model.py --constraint_claim_loss_weight {} --constraint_evidence_loss_weight {} --claim_loss_weight {} --evidence_loss_weight {} --para {} --batch_size 20".format(round(cc, 3), round(ce, 3), claim, evidence, para), file=fout)