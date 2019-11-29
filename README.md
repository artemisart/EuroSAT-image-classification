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

          AnnualCrop      0.926     0.923     0.925       300
              Forest      0.963     0.956     0.959       297
HerbaceousVegetation      0.868     0.934     0.900       302
             Highway      0.861     0.874     0.867       269
          Industrial      0.945     0.934     0.939       257
             Pasture      0.871     0.904     0.887       187
       PermanentCrop      0.900     0.846     0.872       254
         Residential      0.958     0.949     0.954       314
               River      0.895     0.873     0.884       245
             SeaLake      0.978     0.964     0.971       275

            accuracy                          0.918      2700
           macro avg      0.916     0.916     0.916      2700
        weighted avg      0.919     0.918     0.918      2700

Confusion matrix
                      Ann  For  Her  Hig  Ind  Pas  Per  Res  Riv  Sea
AnnualCrop            277    0    2    5    0    4    8    0    4    0
Forest                  0  284    7    0    0    2    0    0    2    2
HerbaceousVegetation    2    4  282    0    0    6    6    0    0    2
Highway                10    1    1  235    4    2    5    2    9    0
Industrial              0    0    2    5  240    0    3    5    2    0
Pasture                 3    5    2    2    0  169    0    2    2    2
PermanentCrop           4    0   22    4    2    4  215    1    2    0
Residential             0    1    4    3    5    0    0  298    3    0
River                   0    0    2   18    3    3    2    3  214    0
SeaLake                 3    0    1    1    0    4    0    0    1  265
```

You can learn about the available options with `predict.py -h`.
