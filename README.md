# Feedback-Analysis-Using-Random-Forest-Classifier

This repository contains the implementation of a feedback analysis model using a Random Forest classifier. The project is divided into three main modules:

1. **Data Collection**
2. **Feature Extraction**
3. **Random Forest Classifier Training**

## Table of Contents

- [Feedback Analysis Model](#feedback-analysis-model)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Modules](#modules)
    - [Module 1: Data Collection](#module-1-data-collection)
    - [Module 2: Feature Extraction](#module-2-feature-extraction)
    - [Module 3: Random Forest Classifier Training](#module-3-random-forest-classifier-training)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

This project aims to analyze customer feedback and determine overall sentiment using machine learning techniques. The feedback data is preprocessed, features are extracted, and a Random Forest classifier is trained to predict the sentiment of the feedback.

## Modules

### Module 1: Data Collection

This module involves gathering relevant data for training and testing the feedback analysis model. The steps are as follows:

- Iterate through all records in the data source.
- Check if each record contains valid feedback.
- Load feedback into memory and standardize its format.
- Append the feedback content and provided label to separate lists.
- Return the lists containing the collected feedback and their corresponding labels.

### Module 2: Feature Extraction

This module involves gathering and analyzing feedback data to determine overall sentiment using a Random Forest classifier and the NLTK library for text preprocessing.

**Steps:**

1. **Collect Data**
   - Ratings (1-5)
   - Recommendation (0 - Not Recommended, 1 - Recommended)
   - Description Value (Length of the feedback description in number of words)
2. **Data Preprocessing**
   - Handle missing values.
   - Normalize or standardize numerical features.
   - Encode categorical variables.
   - Use NLTK to preprocess text data by removing common stopwords.
3. **Feature Extraction**
   - Extract features such as TF-IDF (Term Frequency-Inverse Document Frequency) from the feedback text.
4. **Implement Machine Learning Algorithms**
   - Select and implement algorithms suitable for feedback analysis. In this case, a Random Forest classifier is used.
5. **Train and Evaluate the Model**
   - Train the Random Forest classifier on the preprocessed and feature-extracted feedback data.
   - Evaluate the model's performance to ensure it accurately analyzes feedback sentiment.

### Module 3: Random Forest Classifier Training

This module focuses on training and evaluating the Random Forest classifier.

**Steps:**

- Split data into training (80%) and testing (20%) sets.
- Create a Random Forest Classifier and train it on the training features (`X_train`) and labels (`y_train`).
- Use the trained model to predict the labels for the testing data (`X_test`).
- Evaluate the model's performance by printing a classification report detailing precision, recall, F1-score, and support for each class.
- Calculate and print the overall accuracy of the model on the testing set.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/feedback-analysis.git

Usage
Prepare your data source and place it in the specified path.
Run the data collection module to gather and preprocess the feedback data.
Execute the feature extraction module to process the text data and extract relevant features.
Train the Random Forest classifier using the training module.
Evaluate the model's performance on the testing data.
Results
Below are some sample results showing the performance of the Random Forest classifier.

Output:
`images/Screenshot 2024-06-09 230133.png`
`images/Screenshot 2024-06-09 230001.png`

Contributing
Contributions are welcome! Please read the contributing guidelines first.

License
This project is licensed under the MIT License. See the LICENSE file for details.
