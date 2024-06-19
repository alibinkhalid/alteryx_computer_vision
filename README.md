# Simplifying Image Recognition and Classification Models

## Overview
Even casual viewers can easily recognize characters from "The Simpsons" due to the show's extensive run. This familiarity is akin to the process of training image recognition models with a robust dataset.


## Data Preparation
I used a [Kaggle dataset](https://www.kaggle.com/datasets/mathurinache/simpsons-images) with over 16,000 images from "The Simpsons." For this example, I selected 5,633 images of Homer, Marge, Bart, and Lisa, dividing them into training, validation, and holdout sets at 70%, 20%, and 10%, respectively.

![Workflow Image](https://github.com/alibinkhalid/alteryx_computer_vision/blob/6936c33d7c5dc81fbe03613f71662127ba17ae75/Workflow%20Overview.png)

Consistency in image size is crucial for model training. I standardized images to 128x128 pixels using the Image Processing Tool.

![Image Processing](https://github.com/alibinkhalid/alteryx_computer_vision/blob/71c8e0aa88f3d13d71fc612ff020e6658b81197a/Image%20Processing.png)

## Setting Up Image Recognition
In the Image Recognition Tool, specify the source of training and validation images and the label field. Configure epochs and batch sizes to optimize training.

![Tool Configuration](https://github.com/alibinkhalid/alteryx_computer_vision/blob/61a91d1b443ec7770a13288e7fadaf8cb45ff948/tool%20configuration.png)

# epoch
An epoch is a single pass (forward and backward) of all data in a training set through a neural network. Epochs are related to iterations, but not the same. An iteration is a single pass of all data in a batch of a training set.
Increasing the number of epochs allows the model to learn from the training set for a longer time. But doing that also increases the computational expense.
You can increase the number of epochs to help reduce error in the model. But at some point, the amount of error reduction might not be worth the added computational expense. Also, increasing the number of epochs too much can cause problems of overfitting, while not using enough epochs can cause problems of underfitting.

## Choosing a Pre-trained Model
Select from pre-trained models like InceptionV3, VGG16, InceptionResNetV2, or Resnet50V2, each with distinct trade-offs in accuracy and performance.

Pre-trained models are models that contain feature-extraction methods with parameters that are already defined. Models with more parameters tend to be more accurate, but slower and computationally expensive. The opposite is true for models with fewer parameters; they tend to be less accurate, but faster and computationally cheap.

Here are simplified explanations of the pre-trained models included in the tool. Keep in mind that the performance of these models drastically depends on your data, so the summaries won't always be true.

**VGG16** tends to be the most accurate, slowest, and most computationally expensive. Minimum image size: 32 × 32 pixels.

**InceptionResNetV2** tends to balance accuracy, speed, and computational expense, with some bias toward accuracy. Minimum image size: 75 × 75 pixels.

**Resnet50V2** tends to balance of accuracy, speed, and computational expense, with some bias toward speed and less computational expense. Minimum image size: 32 × 32 pixels.

**InceptionV3** tends to be the least accurate (but still quite accurate), fastest, and least computationally expensive. Minimum image size: 75 × 75 pixels.

Each of those models was trained on a dataset that contained over 14 million images with more than 20,000 labels.

Choosing a pre-trained model allows you to skip training an entire neural network using your own images. When you choose to use a pre-trained model, you're effectively assuming that your input parameters match what the pre-trained model expects, so you don't need to rebuild a model that does about the same thing as the pre-trained one (and might even perform worse). Because many of the features from images tend to be the same as the ones the models have used during training, often you can safely assume that a pre-trained model will work with your input.

Use a pre-trained model when you have images with features that match what the pre-trained model expects and want to avoid training your own model.

## Model Training and Prediction
Execute the workflow and observe the training progress. Typically, accuracy improves and loss decreases over time. Save the trained model in a .yxdb file for future use.

![Prediction Workflow](https://github.com/alibinkhalid/alteryx_computer_vision/blob/ea388ce2e26cecf3993612101e33635652c6380e/Prediction%20Workflow.png)

## Performance Comparison
In tests, Resnet50V2 demonstrated the best balance of accuracy and speed. Here’s a summary of the performance metrics:

| Model              | Training Time | Prediction Time | Accuracy        |
|--------------------|---------------|-----------------|-----------------|
| InceptionV3        | 46 min        | 1 min 44 sec    | 80.46%          |
| InceptionResNetV2  | 1 hr 12 min   | 1 min 28 sec    | 83.84%          |
| VGG16              | 1 hr 54 min   | 3 min 6 sec     | 88.28%          |
| Resnet50V2         | 57 min        | 1 min 28 sec    | 89.88%          |


## Output of Model
![Results](https://github.com/alibinkhalid/alteryx_computer_vision/blob/c47c4aa565c253081c74e222c1286eb030164753/Results.png)


## Practical Applications
Image classification has vast applications. However, always consider [ethical implications](https://www.microsoft.com/en-us/ai/responsible-ai), especially when dealing with personal images.


## Further Reading
- [Convolutional Neural Networks](https://cs231n.github.io)
- [Batch vs. Epoch](https://machinelearningmastery.com)
- [Understanding Accuracy and Loss](https://docs.paperspace.com)





Inspired by:
**Susan Currie Sivek**  
Senior Data Science Journalist
