# deeplearning-ml-challenge
Pandas, deep learning, machine learning, Pandas, 

# **Background**

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

  - **EIN** and **NAME**—Identification columns

  - **APPLICATION_TYPE**—Alphabet Soup application type

  - **AFFILIATION**—Affiliated sector of industry

  - **CLASSIFICATION**—Government organization classification

  - **USE_CASE**—Use case for funding

  - **ORGANIZATION**—Organization type

  - **STATUS**—Active status

  - **INCOME_AMT**—Income classification

  - **SPECIAL_CONSIDERATIONS**—Special considerations for application

  - **ASK_AMT**—Funding amount requested

  - **IS_SUCCESSFUL**—Was the money used effectively

# **Instructions**

**Step 1: Preprocess the Data**

Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

  - Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:

    - What variable(s) are the target(s) for your model?

    - What variable(s) are the feature(s) for your model?

  - Drop the `EIN` and `NAME` columns.

  - Determine the number of unique values for each column.

  - For columns that have more than 10 unique values, determine the number of data points for each unique value.

  - Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

  - Use `pd.get_dummies()` to encode categorical variables.

  - Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

  - Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

**Step 2: Compile, Train, and Evaluate the Model**

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

  - Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

  - Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

  - Create the first hidden layer and choose an appropriate activation function.

  - If necessary, add a second hidden layer with an appropriate activation function.

  - Create an output layer with an appropriate activation function.

  - Check the structure of the model.

  - Compile and train the model.

  - Create a callback that saves the model's weights every five epochs.

  - Evaluate the model using the test data to determine the loss and accuracy.

  - Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

**Step 3: Optimize the Model**

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

  - Use any or all of the following methods to optimize your model:

    - Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

      - Dropping more or fewer columns.

      - Creating more bins for rare occurrences in columns.

      - Increasing or decreasing the number of values for each bin.

      - Add more neurons to a hidden layer.

      - Add more hidden layers.

      - Use different activation functions for the hidden layers.

      - Add or reduce the number of epochs to the training regimen.

**Note:** If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

  - Create a new Google Colab file and name it `AlphabetSoupCharity_Optimization.ipynb`.

  - Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

  - Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

  - Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

  - Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.

**Step 4: Copy Files Into Your Repository**

Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

  - Download your Colab notebooks to your computer.

  - Move them into your Deep Learning Challenge directory in your local repository.

  - Push the added files to GitHub.



# **Report on the Neural Network Model**

**Overview of the analysis:**

The purpose of this analysis is to develop a deep learning model using a feedforward neural network to predict the success of funding applications for Alphabet Soup, a philanthropic organization. The model aims to classify funding applications as successful or unsuccessful based on various input features.

  - **Data Preprocessing**

    - Target Variable(s):

      - The target variable for the model is the binary outcome indicating whether a funding application was successful or not (ie: **IS_SUCCESSFUL**).

    - Features:

      - The features for this model include various input variables such as application type, organization classification, and other relevant factors provided in the dataset.

    - Variables Removed:

      - The variables `EIN`, `NAME`, `SPECIAL_CONSIDERATIONS`, and `STATUS` were removed from the input data as they are neither targets nor features for the model.

  - **Compiling, Training, and Evaluating the Model**

    - Model Architecture:

      - The neural network model consisted of two hidden layers, each with 64 neurons and ReLU activation functions. The output layer had a single neuron for binary classification.

      - ReLU activation functions were chosen for the hidden layers due to their effectiveness in addressing the vanishing gradient problem and speeding up convergence.

    - Model Performance:

      - The model achieved an accuracy of ~73.01% and a mean squared error (MSE) loss of ~0.1859 on the test dataset.

      - While the model's accuracy is "decent", further optimization may be required to meet specific performance targets or improve generalization.

    - Steps to Increase Model Performance:

      - Experimentation was done with different architectures, varying the number of neurons, layers, or activation functions.

      - Tuning hyperparameters such as learning rate, batch size, and dropout rate to improve convergence and prevent overfitting took place.

      - Feature engineering, including selecting relevant features, encoding categorical variables effectively, and handling missing values appropriately increased performance.

  - **Summary**

The deep learning model was developed by using a feedforward neural network which demonstrated reasonable performance in predicting the success of funding applications for Alphabet Soup. However, further optimization and fine-tuning would be necessary to achieve higher accuracy and better generalization on unseen data.

For a different approach to solving this classification problem, a convolutional neural network (CNN) could be considered. CNNs are well-suited for tasks involving image data or sequential data, such as text classification. By leveraging the spatial hierarchies present in the input data, CNNs could capture intricate patterns and relationships, potentially improving classification accuracy. Additionally, techniques like transfer learning, where pre-trained CNN models are fine-tuned on the specific dataset, could be explored to leverage the knowledge learned from large-scale datasets. Overall, a CNN may offer better performance and generalization for the classification problem at hand.

# **Citations:**

IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site.

Instructor: [Othmane Benyoucef](https://www.linkedin.com/in/othmane-benyoucef-219a8637/)

Model Suggestion: [Michael Flesch](https://www.linkedin.com/in/michael-flesch/)
