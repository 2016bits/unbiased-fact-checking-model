export CUDA_VISIBLE_DEVICES=0
python debias/unbiased_model.py --constraint_loss_weight 0.001
python debias/unbiased_model.py --constraint_loss_weight 0.002
python debias/unbiased_model.py --constraint_loss_weight 0.003
python debias/unbiased_model.py --constraint_loss_weight 0.004
python debias/unbiased_model.py --constraint_loss_weight 0.005
python debias/unbiased_model.py --constraint_loss_weight 0.006
python debias/unbiased_model.py --constraint_loss_weight 0.007