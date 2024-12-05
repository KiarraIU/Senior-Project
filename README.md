# Building a Statistical Model to Ensure Accurate and Early Alzheimer’s Detection
The primary focus of this project is multi-class classification. The goal was to create a machine learning model that is able to classify MRI images into the stages of Alzheimer's: MildDemented, ModerateDemented, NonDemented, and VeryMildDemented. The model's is as follows:
- Use pre-trained Xception architecture to serve as the model base and additional classification layers are implemented to improve accuracy. An ImageDataGenerator was used to implement class weight balancing, dropout, and early stopping, so that the model does not become overfitted and the performance improves. 


Image Directories: The dataset is stored in four directories representing different stages of Alzheimer's disease.
- MildDemented_Dir
- ModerateDemented_Dir
- NonDemented_Dir
- VeryMildDemented

 
## Installation Requirements 
To get started, you need to install the requried dependencies. 
- os
- pandas
- keras
- matplotlib
- seaborn
- scikit-learn
- numpy
- tensorflow
  
The dependencies are installed using pip. 
pip install os pandas keras matplotlib seaborn scikit-learn numpy tensorflow


## Dataset
Image Directories: The dataset is stored in four directories representing different stages of Alzheimer's disease.
- MildDemented_Dir
- ModerateDemented_Dir
- NonDemented_Dir
- VeryMildDemented

  
You can download the dataset from [https://www.kaggle.com/datasets/enasemad/alzheimer1]. The dataset should be organized into separate directories for each class, which will be used for training, validation, and testing.


## Model Architecture 
The model uses the Xception architecture, but it is going to be pre-trained on ImageNet as a base. The top layers of the model are removed so that parameter optimization can occur. Custom layers are added to the model to assist in classification, such as the dropout layers, dense layers, and output layers. The dense are critical for feature learning. 


## Functions
### create_gen_img(train_set, test_images, val_set, batch_size)
Purpose:
Creates image data generators for the training, validation, and testing datasets.

Inputs:
- train_set: Path to the training dataset.
- test_images: Path to the testing dataset.
- val_set: Path to the validation dataset. 
- batch_size: The number of images to process within each of the batches.

Outputs:
- train_gen: Generator for the training images.
- valid_gen: Generator for the validation images.
- test_gen: Generator for the testing images.


### show_images(gen)
This function displays a grid of 25 augmented images that are produces using an ImageDataGenerator. 

Input:
- gen: An image generator

Output:
- A 5x5 grid of augmented MRI images with the corresponding class labels.


### train_model(train_gen, valid_gen, epochs)
Trains the model on the training dataset while using the validation dataset to monitor the performance. \

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

Once the model is trained, you can use it to make predictions on the new MRI images. To predict the label of a new image, the image is passed through the model and the output is decoded. 


## Contributions 

Contributions are not welcome at this time. 


## License 

This project is licensed under the MIT License - see the LICENSE file for details.

### 
