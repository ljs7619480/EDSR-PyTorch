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

#VRDLx3_MSE
# python3 main.py --epochs 100 \
#                 --scale 3 \
#                 --patch_size 15 \
#                 --data_range '1-285/286-291' \
#                 --decay "10-20-30-40-100-200" \
#                 --save "VRDLx3_MSE" \
#                 --pre_train "../experiment/VRDLx3/model/model_best.pt" \
#                 --loss '1*MSE'

# VRDLx3 + dilated convolution
# python3 main.py --epochs 300 \
#                 --scale 3 \
#                 --patch_size 15 \
#                 --data_range '1-285/286-291' \
#                 --decay "10-20-30-40-100-200" \
#                 --save "VRDLx3_Dila" \
#                 --dilation

# VRDLx3 + unknown_interpolation
# python3 main.py --epochs 300 \
#                 --scale 3 \
#                 --patch_size 15 \
#                 --data_range '1-285/286-291' \
#                 --decay "10-20-30-40-100-200" \
#                 --save "VRDLx3_UNK" \
#                 --pre_train "../experiment/VRDLx3/model/model_best.pt" \
#                 --data_postfix 'unknown'

# L2, Unknown + Dilation(ALL)
# python3 main.py --epochs 300 \
#                 --scale 3 \
#                 --patch_size 15 \
#                 --data_range '1-285/286-291' \
#                 --decay "10-20-30-40-100-200-300-400-500-600-700-800-900" \
#                 --save "VRDLx3_ALL" \
#                 --loss '1*MSE' \
#                 --pre_train "../experiment/VRDLx3/model/model_best.pt" \
#                 --data_postfix 'unknown' \
#                 --dilation

# L1, Unknown + Dilation
python3 main.py --epochs 300 \
                --scale 3 \
                --patch_size 15 \
                --data_range '1-285/286-291' \
                --decay "10-20-30-40-100-200-300-400-500-600-700-800-900" \
                --save "VRDLx3_Dila_UNK" \
                --pre_train "../experiment/VRDLx3/model/model_best.pt" \
                --data_postfix 'unknown' \
                --dilation


# --self_ensemble