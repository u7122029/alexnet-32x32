# alexnet-32x32
A modified version of AlexNet that modifies the first convolutional layer so that not all features are lost from a 32x32x3 image before reaching the fully-connected layers.

This is a different approach to keeping the model completely unchanged, but resizing each image to 256x256, then centre cropping them to vit the 224x224x3 dimensions that AlexNet normally takes.
## Model Structure
Our model uses the same architecture from https://pytorch.org/hub/pytorch_vision_alexnet/, but since we are not enlargening the 32x32 images to 224x224, we make the following modifications:
1. Change the kernel size of the first convolutional layer to 3x3 and the stride to 1 to ensure that the outputted tensor is no longer too small by the time it reaches the classification layers.
2. Set the padding of the first convolutional layer to 1 to ensure that the outputted tensor has the same width and height as when it was inputted - also helps ensure that the tensor does not become too small before reaching the classification layers.
3. Remove the 2d adaptive average pooling layer because the size of the tensor would be 3x3 by the time it reaches this layer. Downsampling would lose overly many features, and upsampling does not help with learning features.
## Training
The original paper uses the SGD optimiser, but we use Adam for faster convergence and adaptive momentum.
- Learning rate: `5e-5`,
- Weight decay: `0.001`
## Results
