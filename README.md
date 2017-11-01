# WeightedLoss_Convolutional_Pose_Machines_PyTorch

PyTorch (re)implement of Convolutional Pose Machine [1] with weighted loss as an option. More information about weighted loss please refer to [another repo of me](https://github.com/JindongJiang/WeightedLoss_DeepPose_PyTorch).

Please notice that this is an underdeveloped implementation of Convolutional Pose Machine for my own experiment. But the architecture and training are very close to the original model, please feel free to use it for your own project.

## Prerequisites
* Python 3.6
* scipy
* sklearn
* pillow
* PyTorch 0.2
* torchvision 0.1.9
* tensorboardX (only if you need tensorboard summary)
* TensorFlow (for tensorboard web server)
* OpenCv > 3.0

## Download datas
I found that original link of the Leeds Sports Pose Dataset at University of Leeds has been removed. You can download the dataset [here](http://sam.johnson.io/research/lsp.html) and the extended dataset [here](http://sam.johnson.io/research/lspet.html).

Please download the dataset and unzip it in `data` folder with a directory tree like this:

```bash
data
└── LSP
    ├── lsp_dataset
    │   ├── images
    │   └── visualized
    └── lspet_dataset
        └── images
```

## Usage
### Training
#### With weighted loss
```bash
python -W ignore::UserWarning cpm_train.py --lsp-root ./data/LSP --ckpt-dir ./model  --summary-dir ./summary --cuda
```
#### Without weighted loss
```bash
python -W ignore::UserWarning cpm_train.py --lsp-root ./data/LSP --ckpt-dir ./model  --summary-dir ./summary --cuda --wl
```
More argument for training please refer to `cpm_train.py`.

## References

[1] [Wei, Shih-En, et al. "Convolutional pose machines." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.](https://arxiv.org/abs/1602.00134)
