# Early Alzheimer's Detection through MRI Images
## Overview
This project focuses on building a machine learning model to classify MRI images of patients into four categories: MildDemented, ModerateDemented, NonDemented, and VeryMildDemented. The model uses a convolutional neural network (CNN) based on the Xception architecture, with additional layers for classification. The model is trained using a data generator and employs techniques such as class weight balancing, dropout, and early stopping to improve performance and prevent overfitting.

## Project Structure 
1. Installation
2. Dataset
4. Model Architecture
5. Training
6. Evaluation
7. Usage
8. Contributing
9. License

Image Directories: The dataset is stored in four directories representing different stages of Alzheimer's disease.
- MildDemented_Dir
- ModerateDemented_Dir
- NonDemented_Dir
- VeryMildDemented

## Installation 
To get started, you need to install the requried dependencies. 
### Requirements 
- os
- pandas
- keras
- matplotlib
- seaborn
- scikit-learn
- numpy
- tensorflow

Install these dependencies using pip.
pip install os pandas keras matplotlib seaborn scikit-learn numpy tensorflow

## Dataset
The dataset used for this model contains MRI images with labels representing the different stages of Alzheimer's disease: 
- MildDemented_Dir
- ModerateDemented_Dir
- NonDemented_Dir
- VeryMildDemented
- 
You can download the dataset from [https://www.kaggle.com/datasets/enasemad/alzheimer1]. The dataset should be organized into separate directories for each class, which will be used for training, validation, and testing.

## Model Architecture 
The model uses the Xception architecture pre-trained on ImageNet as a base. The top layers of the model are removed to allow for fine-tuning. Custom layers are added on top for classification, including dropout layers to prevent overfitting, dense layers for feature learning, and an output layer with softmax activation to classify the images into the four categories.

## Functions
### create_gen_img(train_set, test_images, val_set, batch_size)
Purpose:
Creates image data generators for the training, validation, and testing datasets.
Inputs:
- train_set: Path to the training dataset.
- test_images: Path to the testing dataset.
- val_set: Path to the validation dataset. 
- batch_size: The number of images to process in each batch.

Outputs:
- train_gen: Generator for the training images.
- valid_gen: Generator for the validation images.
- test_gen: Generator for the testing images.

### show_images(gen)
This function displays a grid of 25 augmented images that were produced by an ImageDataGenerator.
Input:
- gen: An image generator

Output:
- A 5x5 grid of augmented MRI images with the corresponding class labels.

### train_model(train_gen, valid_gen, epochs)
Trains the model on the training dataset while using the validation dataset to monitor the performance. 
Inputs:
- train_gen: Training data generator
- valid_gen: Validation data generator
- epochs: Number of training epochs

Outputs:
- model: Trained CNN model

### evaluate_model(model, test_gen)
Evaluates the training model on the testing dataset.
Inputs:
- model: Trained CNN model
- test_gen: Training data generator

Outputs:
- accuracy:Classification accuracy of the model on the testing data 
- loss: Loss metric of the model on the testing data

## Training
To train the model, the images are loaded using an image data generator with data augmentation. The model uses class weights to address class imbalances and early stopping to prevent overfitting. The training process is as follows:
1. Dataset Preparation: Organizing the MRI images into directories and splitting them into training, validation, and testing sets.
2. Data Augmentation: Create altered image batches using the create_gen_img() function.
3. Model Training: Use train_model() function with the data generators. 


## Evaluation
After training, the model's performance is evaluated on the testing dataset. The evaluation metrics include accuracy, loss, and a confusion matrix.
1. Use the evaluate_model() function to evaluate performance

## Usage 
Once the model is trained, you can use it to make predictions on new MRI images. To predict the label of a new image, you can pass the image through the model and decode the output.

## Contributions 
Contributions are welcome! If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License 
This project is licensed under the MIT License - see the LICENSE file for details.

### 
