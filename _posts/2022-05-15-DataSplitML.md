# What is TTV split in Machine Learning?
T: Training, T: Testing, V: Validation.

Data splitting is a crucial step in machine learning that involves dividing a dataset into separate subsets for training, testing, and model optimization. 
The proper allocation of data ensures that the model learns from a diverse range of examples, generalizes well to unseen data, and can be effectively evaluated for performance. In this blog post, I will discuss how to split data into training, testing, and optimization categories, including the required amount of data in each split and the significance of data size for training the model.

1. Understanding the Data Split:
The standard approach for data splitting involves creating three distinct subsets: the training set, the testing set, and the validation set (sometimes referred to as the optimization set). 
Each subset serves a specific purpose in the machine learning pipeline:

  1. Training Set: This subset is used to train the model. It contains labeled examples on which the model learns patterns and makes predictions. 
  The training set should be large enough to capture the underlying data distribution and provide sufficient diversity for effective learning.
  2. Testing Set: The testing set is used to assess the model's performance and generalization ability. 
  It consists of labeled examples that the model has not seen during training. Evaluating the model on the testing set helps estimate its performance on new, unseen data.
  3. Validation Set (Optimization Set): The validation set is optional but highly recommended. 
  It serves as an intermediate set used for model optimization, hyperparameter tuning, and monitoring training progress. 
  It helps prevent overfitting by providing an unbiased evaluation of the model's performance during training.

2. Determining the Data Split Ratio:
    The allocation of data between the training, testing, and validation sets depends on several factors, including the dataset size, the complexity of the problem, and the available computational resources. 
    Here are some general guidelines for data split ratios:

  1. Training Set: The training set typically requires the largest portion of the data, ranging from 60% to 80% of the total dataset. 
  A larger training set allows the model to learn more effectively, especially for complex tasks and deep learning models.
  2. Testing Set: The testing set usually receives a smaller portion, typically around 20% to 30% of the total dataset. 
  This ensures a sufficient number of examples for evaluating the model's performance and assessing its generalization capability.
  3. Validation Set: The validation set, if used, can vary in size. It is often created by splitting a small percentage (e.g., 10% to 20%) of the training set. 
  The size should be large enough to provide a reliable evaluation of the model's performance and aid in hyperparameter tuning.

3. Impact of Data Size on Model Training:
  The amount of data available for model training plays a significant role in achieving good performance. The following considerations should be taken into account:

  1. Sufficient Training Data: Adequate training data is crucial for the model to capture the underlying patterns and generalize well. 
  Insufficient training data can lead to overfitting, where the model memorizes the training examples without learning the underlying concepts. 
  As a rule of thumb, aim for a training set size that captures the data's diversity and complexity.
  2. Trade-off with Model Complexity: The model's complexity, such as the number of parameters, affects the amount of data required for training. More complex models generally require more data to avoid overfitting. 
  Simple models may achieve good performance with smaller datasets, while complex models may need larger datasets to effectively learn the underlying patterns.
  3. Consideration for Imbalanced Data: In cases where the dataset is imbalanced (unequal class distribution), it is essential to ensure that each subset (training, testing

## What does this mean for my assignment?
In the 00 fastai notebook supplied to us, a data splitter was already generated that we could use to split our data into training, testing and optimisation. I chose to have 60% training, 20% testing and 20% optimisation/validation. The split can be completed within the datablock. This is for data sets that are not already split, similar to our Q2, where we had to obtain our own data for the data set then subsequently split it into its own sub data sets. For Q3 the CIFAKE library/ data given already is split into training and testing. In this case we weren't supplied a validation data set as for the kaggle competition to be won, a validation data set is used on the model.<br>

An example of the code used. Here the splitter function within the datablock allows the ability to split the data into their respective categories.
`dls = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, 
splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label, item_tfms=[Resize(192, method='squish')]
).dataloaders(Path('cat_or_not_train')) `
