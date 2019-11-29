# EuroSAT image classification

I decided to use Pytorch as it seemed appropriate and I have more experience
with this framework.  For loading and handling the dataset I choose to implement
a custom loader (subclassing torchvision ImageFolder) to integrate nicely with
pytorch pipelines (e.g. multithreaded data loaders, transform operations,
samplers...).

As EuroSAT does not define a test set, I used a 90/10 split with a fixed random
seed (42) to generate it consistently.  In the [EuroSAT
paper](https://arxiv.org/abs/1709.00029) the authors found that the best
performing model is ResNet-50 among the ones they tried so I used this one as
well (note that this is easily changed).

## Training

The dataset is further splitted for training and validation (so in total we have
train/val/test: 81/9/10).  First the mean and std is computed on each channel
across the train set to normalize the images (crucial if we want to use models
pretrained on ImageNet, and a good idea anyway), and this normalization will be
saved with the model weights.  I choosed to use tensorboard for logging as I
find it much more productive than printing to console both for analysis and
comparison between models/runs/parameters.

For the training I use a bit of data augmentation with random horizontals &
verticals flips (available with torchvision utilities), and a pretrained
resnet50 model (with its head replaced as we only have 10 classes).  By default
only the head is finetuned.  The script can also train a model from scratch
(`--pretrained no`).

After each epoch, the accuracy is computed on the validation set and the best
model of the current run is saved in `weights/best.pt` (WARNING: it overwrites
the checkpoints from previous runs).

```sh
> tensorboard --logdir runs
> ./train.py
...
New best validation accuracy: 0.8839505910873413
...
Epochs: 100% 10/10 [03:02<00:00, 18.14s/it]
```

You can learn about the available options with `train.py -h`.

## Inference

The script `predict.py` performs inference.  It loads a model checkpoint (`-m`,
by defaults `weights/best.pt`) and runs it on the specified files:
```sh
> ./predict.py one_image.png and_a_folder/*.jpg
'one_image.png', 0
'and_a_folder/0.jpg', 2
'and_a_folder/1.jpg', 1
...
```
I choose to output the results in csv format so as to easily integrate this
script in any pipeline.  tqdm may show a progress bar but it uses stderr so
doesn't affect the output.

To get a performance report on the test set of EuroSAT, just omit the
files option:
```sh
> ./predict.py
Predict: 100%|██████████████████████████████████████████████████| 43/43 [00:22<00:00,  2.47batch/s]
Classification report
                      precision    recall  f1-score   support

          AnnualCrop      0.880     0.930     0.904       300
              Forest      0.973     0.835     0.899       297
HerbaceousVegetation      0.775     0.911     0.837       302
             Highway      0.791     0.814     0.802       269
          Industrial      0.950     0.895     0.922       257
             Pasture      0.851     0.824     0.837       187
       PermanentCrop      0.909     0.748     0.821       254
         Residential      0.934     0.949     0.942       314
               River      0.816     0.816     0.816       245
             SeaLake      0.900     0.982     0.939       275

            accuracy                          0.875      2700
           macro avg      0.878     0.870     0.872      2700
        weighted avg      0.879     0.875     0.875      2700

Confusion matrix
                      Ann  For  Her  Hig  Ind  Pas  Per  Res  Riv  Sea
AnnualCrop            279    0    3    4    0    0    5    0    6    3
Forest                  3  248   21    4    0    9    0    0    2   10
HerbaceousVegetation    3    4  275    2    0    7    5    1    2    3
Highway                12    1    1  219    3    3    6    4   20    0
Industrial              1    0    1    9  230    0    0    9    4    3
Pasture                 5    2   10    1    0  154    3    2    4    6
PermanentCrop           7    0   34    6    3    5  190    3    4    2
Residential             0    0    6    4    3    0    0  298    3    0
River                   4    0    3   28    3    2    0    2  200    3
SeaLake                 3    0    1    0    0    1    0    0    0  270
```

You can learn about the available options with `predict.py -h`.
