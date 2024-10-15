import numpy as np

fout = open("./run_scripts/unbiased_scaled.sh", 'w')
print("export CUDA_VISIBLE_DEVICES=2", file=fout)

for scaled in np.arange(0.2, 2.1, 0.2):
    print("python two_class_scripts/unbiased_model_emplify_bias.py --scaled_rate {}".format(round(scaled, 3)), file=fout)