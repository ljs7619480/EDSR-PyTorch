# Overview
In this project, I try to train a [Enhanced Deep Residual Networks for Single Image Super-Resolution(EDSR)](https://github.com/thstkdgus35/EDSR-PyTorch) model on my [dataset](https://drive.google.com/drive/u/3/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x).

In the beginning, I try to write a new dataloader to load my own training data. However, I found this project has very few comments and documents about this project and it's a little cost time to understand how this project organized. Thankfully, the project is default training on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset, and thus, I turn to implement a simple converter to convert my dataset to DIV2K format.
```
DIV2K/
├── DIV2K_test_lr_unknown
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   ├── X2
│   ├── X3
|   └── ...
└── DIV2K_train_LR_unknown
    ├── X2
    ├── X3
    └── ...
```

Here shows the training result.
| model           | LR      | Loss_fn | self-ensemble | score  |
|-----------------|---------|---------|---------------|--------|
| EDSRx3          | Bicubic | L1-norm |       X       | 25.289 |
| EDSRx3          | Bicubic | L1-norm |       V       | 25.395 |
| EDSRx3          | Bicubic | L2-norm |       V       |  |
| EDSRx3+Dilation | Bicubic | L1-norm |       V       | 25.439 |
![img img img img img img]()

# Prepare data
1. Download the training data and testing data [here](https://drive.google.com/drive/u/3/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x).
2. move your training data under `datasets/source` folder.
```
datasets
└── source
    ├── testing_lr_images
    └── training_hr_images
```
3. run `pre_process.py`
```
$ cd datasets/
$ python3 pre_process.py [--help]
```
4. Now, the dataset should be organized as below
```
datasets
├── DIV2K
│   ├── DIV2K_test_lr_unknown
│   ├── DIV2K_train_HR
│   ├── DIV2K_train_LR_bicubic
│   │   ├── X2
│   │   └── X3
│   └── DIV2K_train_LR_unknown
│       ├── X2
│       └── X3
└── source
    ├── testing_lr_images
    └── training_hr_images
```
Note: The folder name is a little confusing. It is because the dataloader of this project use path name to specify which dataloader module shoud be used. Thus, I follow its name rule.

# Training 
- I did not modify the config file `src/option.py` intended I setup my configuration by command line parameters, and you can find all my training script in `src/demo.py`
- You can directly run the script by
    ``` 
    $ cd src; chmod +x train.sh; ./train.sh
    ```
    , and then, all your training result can be found in `experiments` folder including the log file.


# Prediction
Run the following command and your testing result will be save to `experiment/<name>`
```
$ python3 main.py --data_test Demo --dir_demo <path_to_testing_imgs> --scale 3 \
        --pre_train <path_to_model_weight> --test_only --save_results --save <name>
```
