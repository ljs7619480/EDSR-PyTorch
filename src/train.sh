# data_range: train_idx_start-train_idx_end/val_start_idx-val_start_idx'
# decay: scheduler.MultiStepLR.milestones
# reset: rm -rf exeriment/args.save

# VRDLx2
# python3 main.py --epochs 40 \
#                 --scale 2 \
#                 --patch_size 20 \
#                 --data_range "1-285/286-291" \
#                 --decay "10-20-30-40" \
#                 --save "VRDLx2"

#VRDLx3
# python3 main.py --epochs 300 \
#                 --scale 3 \
#                 --patch_size 15 \
#                 --data_range '1-285/286-291' \
#                 --decay "10-20-30-40-100-200" \
#                 --save "VRDLx3" \
#                 --pre_train "../experiment/VRDLx2/model/model_best.pt"

#VRDLx3 + dilated convolurion
python3 main.py --epochs 300 \
                --scale 3 \
                --patch_size 15 \
                --data_range '1-285/286-291' \
                --decay "10-20-30-40-100-200" \
                --save "VRDLx3_Dila" \
                --dilation

# --self_ensemble