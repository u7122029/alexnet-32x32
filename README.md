# alexnet-32x32
A modified version of AlexNet that modifies the first convolutional layer so that not all features are lost from a 32x32x3 image before reaching the fully-connected layers.

This is a different approach to keeping the model completely unchanged, but resizing each image to 256x256, then centre cropping them to vit the 224x224x3 dimensions that AlexNet normally takes.

We train the model using pytorch-lightning 