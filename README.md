# cycleGAN-Implementation

Our own implementation of cycleGAN (https://junyanz.github.io/CycleGAN/).

## Testing

For testing, run test.py with the following flags:

* --

## Training

For training, run train.py with the following flags:

* --name the name for the new model, e.g. test_model
* --dataroot folder with the dataset, should have subfolders trainA and trainB
* --preprocess default is crop, where random patches of 256x256 (can be changed with --crop_size) are cropped from the images and used for training. Other possibility is none, where the dataloader only ensures that the size of the image is divisible by four.

To continue training from the latest version of a model, add --load_model. The epoch where training should resume can be set with --epoch.
