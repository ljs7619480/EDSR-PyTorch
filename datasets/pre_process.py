import cv2
import os
import argparse
import random
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--src_train', default='source/training_hr_images',
                        help="Path to high resolution training images")
    parser.add_argument('--src_test', default='source/testing_lr_images',
                        help="Path to high resolution testing images")
    parser.add_argument('--file_ext', default='.png',
                        help="Image file extension leading with '.'")
    parser.add_argument('--HRs_train', default='DIV2K/DIV2K_train_HR',
                        help="Path to store high resolution training image copies")
    parser.add_argument('--HRs_test', default='DIV2K/DIV2K_test_lr_unknown',
                        help="Path to store high resolution testing image copies")
    parser.add_argument('--LRs_cub', default='DIV2K/DIV2K_train_LR_bicubic',
                        help="Path to store output low resolution training images")
    parser.add_argument('--LRs_unk', default='DIV2K/DIV2K_train_LR_unknown',
                        help="Path to store output low resolution training images")
    parser.add_argument('--scales', default='X2-X3',
                        help="training scale, leading by 'X', and seperate by '-'")
    parser.add_argument('--fit-test', action='store_true',
                        help="adding testing image to train data")
    args = parser.parse_args()
    args.scales = args.scales.split('-')
    args.inter_set = [cv2.INTER_CUBIC, cv2.INTER_NEAREST,
                      cv2.INTER_AREA, cv2.INTER_LINEAR]
    return args


def build_data(src_dir, cpy_dir, cub_dir, unk_dir, scales, save_lr):
    for img_name in tqdm(os.listdir(src_dir)):
        img_HR = cv2.imread(os.path.join(src_dir, img_name))
        h, w, _ = img_HR.shape
        # save img_HR
        cv2.imwrite(os.path.join(cpy_dir, img_name), img_HR)
        if save_lr:
            # save img_LR_cub
            for scale in scales:
                div = int(scale[1:])
                img_LR = cv2.resize(img_HR, (w//div, h//div),
                                    interpolation=cv2.INTER_CUBIC)
                new_name = img_name.replace(
                    args.file_ext, scale.lower()+args.file_ext)
                cv2.imwrite(os.path.join(cub_dir, scale, new_name), img_LR)
            # save img_LR_unk
            for scale in scales:
                div = int(scale[1:])
                img_LR = cv2.resize(img_HR, (w//div, h//div),
                                    interpolation=random.choice(args.inter_set))
                new_name = img_name.replace(
                    args.file_ext, scale.lower()+args.file_ext)
                cv2.imwrite(os.path.join(unk_dir, scale, new_name), img_LR)


if __name__ == "__main__":
    args = get_args()
    print(args)

    for s in args.scales:
        os.makedirs(os.path.join(args.LRs_cub, s), exist_ok=True)
        os.makedirs(os.path.join(args.LRs_unk, s), exist_ok=True)
    os.makedirs(args.HRs_train, exist_ok=True)
    os.makedirs(args.HRs_test, exist_ok=True)

    print(">>>>>>>> \nprocess training data <<<<<<<<")
    build_data(args.src_train, args.HRs_train,
               args.LRs_cub, args.LRs_unk, args.scales, True)

    print(">>>>>>>> \nprocess testing data <<<<<<<<")
    build_data(args.src_test, args.HRs_test,
               args.LRs_cub, args.LRs_unk, args.scales, args.fit_test)
