# Early Alzheimer's Detection through MRI Images
This project provides code for loading, preprocessing, and displaying image data from an Alzheimer's disease dataset. The dataset consists of images categorized into four classes: Mild Demented, Moderate Demented, Non-Demented, and Very Mild Demented. The code helps in splitting the dataset, generating image batches, and visualizing the images.

## Project Structure 
Image Directories: The dataset is stored in four directories representing different stages of Alzheimer's disease.
- MildDemented_Dir
- ModerateDemented_Dir
- NonDemented_Dir
- VeryMildDemented

## Requirements
- os
- pandas
- matplotlib
- numpy
- tensorflow

Install these libraries using pip.
pip install os pandas matplotlib numpy tensorflow

## Functions and Usage
split_data(dict_list)
Inputs:
- dict_list: A list of directory paths for each class
Outputs:
train_df: Dataframe of 80% of the data.
valid_df: Dataframe of 10% of the data.
test_df: Dataframe of 10% of the data.

create_gen_img(train_df, valid_df, test_df, batch_size)
Inputs:
- train_df, valid_df, test_df
- batch_size: The number of images to load in each batch.
show_images(gen)
This function displays a grid of 25 augmented images that were produced by an ImageDataGenerator.
Input:
- gen: An image generator
Output:
- A 5x5 grid of augmented MRI images with the corresponding class labels. 














