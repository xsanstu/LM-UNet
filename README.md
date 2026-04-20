# LM-UNet
Code for the Paper "A Lightweight UNet with Mamba for Precipitation Nowcasting"

Basically, only the following requirements are needed:
```
tqdm
torch
lightning
tensorboard
torchsummary
h5py
numpy
```

---
For the paper we used the [Lightning](https://github.com/Lightning-AI/lightning) -module (PL) which simplifies the training process and allows easy additions of loggers and checkpoint creations.
In order to use PL we created the model [UNetDS_Attention](models/unet_precip_regression_lightning.py) whose parent inherits from the pl.LightningModule. This model is the same as the pure PyTorch SmaAt-UNet implementation with the added PL functions.

### Training
An example [training script](train_SmaAtUNet.py) is given for a classification task (PascalVOC).

For training on the precipitation task we used the [train_precip_lightning.py](train_precip_lightning.py) file.
The training will place a checkpoint file for every model in the `default_save_path` `lightning/precip_regression`.
After finishing training place the best models (probably the ones with the lowest validation loss) that you want to compare in another folder in `checkpoints/comparison`.

### Testing
The script [calc_metrics_test_set.py](calc_metrics_test_set.py) will use all the models placed in `checkpoints/comparison` to calculate the MSEs and other metrics such as Precision, Recall, Accuracy, F1, CSI, FAR, HSS.
The results will get saved in a json in the same folder as the models.

The metrics of the persistence model (for now) are only calculated using the script [test_precip_lightning.py](test_precip_lightning.py). This script will also use all models in that folder and calculate the test-losses for the models in addition to the persistence model.
It will get handled by [this issue](https://github.com/HansBambel/SmaAt-UNet/issues/28).

### Precipitation dataset
The dataset consists of precipitation maps in 5-minute intervals from 2016-2019 resulting in about 420,000 images.

The dataset is based on radar precipitation maps from the [The Royal Netherlands Meteorological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about).
The original images were cropped as can be seen in the example below:
![Precip cutout](Precipitation%20map%20Cutout.png)

The dataset is already normalized using a [Min-Max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).