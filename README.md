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

## Choosing a Model
Select from pre-trained models like InceptionV3, VGG16, InceptionResNetV2, or Resnet50V2, each with distinct trade-offs in accuracy and performance.


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


##Output of Model
[Results](https://github.com/alibinkhalid/alteryx_computer_vision/blob/c47c4aa565c253081c74e222c1286eb030164753/Results.png)


## Practical Applications
Image classification has vast applications. However, always consider ethical implications, especially when dealing with personal images.


## Further Reading
- [Convolutional Neural Networks](https://cs231n.github.io)
- [Batch vs. Epoch](https://machinelearningmastery.com)
- [Understanding Accuracy and Loss](https://docs.paperspace.com)





Inspired by:
**Susan Currie Sivek**  
Senior Data Science Journalist
