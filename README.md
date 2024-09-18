# Car-Price-Prediction-using-PyTorch
This project aims to build a Car Price Prediction model using PyTorch. The model predicts the Manufacturer's Suggested Retail Price (MSRP) of cars based on several features such as brand, model, engine specifications, and more. The dataset used for training is from Kaggle and contains various car features and prices.
Project Overview
In this project, we utilize deep learning techniques, particularly Convolutional Neural Networks (CNN) and fully connected layers, to predict car prices based on their specifications. While CNNs are typically used for image data, we adapt the architecture to handle structured data in this case.

Features Used:
Make: Car manufacturer (e.g., Toyota, BMW)
Year: Year of manufacture
Engine HP: Engine horsepower
Engine Cylinders: Number of engine cylinders
Transmission Type: Automatic or manual
Driven Wheels: Front-wheel drive, rear-wheel drive, or all-wheel drive
Number of Doors: Number of doors on the vehicle
The model is trained on a dataset containing these features and uses PyTorch for building and training the model.

Dataset
The dataset used in this project is the Car Features and MSRP dataset from Kaggle. It contains data on car specifications along with the MSRP (price). The dataset has been pre-processed to handle missing values and categorical features using encoding techniques like Label Encoding.

https://www.kaggle.com/datasets/CooperUnion/cardataset/data

Model Architecture
The model consists of a simple fully connected neural network (FCNN) with the following architecture:

Input Layer: 7 features (Make, Year, Engine HP, etc.)
Hidden Layer 1: 128 neurons
Hidden Layer 2: 64 neurons
Output Layer: 1 neuron (price prediction)
Loss Function:
Mean Squared Error (MSE) is used to minimize the difference between predicted and actual car prices.
Optimizer:
The model is optimized using Adam optimizer with a learning rate of 0.001.
Results
The model is trained over 50 epochs. The Mean Squared Error (MSE) on the test set is used to evaluate performance. Additional improvements could be made through hyperparameter tuning, feature engineering, or more complex architectures.

Future Enhancements
Model Improvement: Explore more complex architectures like gradient boosting or using pre-trained embeddings for the categorical features.
Deployment: Create a web interface using Flask to allow users to input car specifications and predict prices in real-time.
