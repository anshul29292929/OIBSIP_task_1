# OIBSIP_task_1
# Iris Flower Classification with K-Nearest Neighbors

This Python program demonstrates how to perform Iris flower classification using the K-Nearest Neighbors (KNN) algorithm. The Iris dataset is a popular dataset for classification tasks and consists of three species of Iris flowers with measurements of their sepal and petal lengths and widths.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Algorithm](#algorithm)
- [Results](#results)
- [License](#license)

## Prerequisites

To run this program, you need to have the following libraries installed:

- NumPy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
Installation
Clone this repository to your local machine:
bash
Copy code
git clone https://github.com/anshul29292929/iris-flower-classification.git
Navigate to the project directory:
bash
Copy code
cd iris-flower-classification
Place the 'Iris.csv' dataset file in the same directory as the Python script.
Usage
Run the Python script iris_classification.py to perform the Iris flower classification using the K-Nearest Neighbors algorithm. You can execute it using a Python interpreter:

bash
Copy code
python iris_classification.py
The script will load the Iris dataset, preprocess the data, train a KNN classifier, make predictions, and display the accuracy of the model.

Dataset
The dataset used in this project is the Iris dataset, which is included in the sklearn.datasets module. It contains the following columns:

Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
Species (the target variable, with three classes: Setosa, Versicolor, and Virginica)
Algorithm
The algorithm used for classification is the K-Nearest Neighbors (KNN) algorithm. KNN is a simple and effective classification algorithm that classifies data points based on their similarity to nearby data points. In this case, it classifies Iris flowers into one of the three species based on their sepal and petal measurements.

Results
After running the program, you will see the accuracy of the KNN classifier on the test data. Additionally, the program generates pairplots to visualize the relationships between features in the Iris dataset.
