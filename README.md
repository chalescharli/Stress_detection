# Stress Detection using Machine Learning

## Overview
This project implements a stress level detection system using multiple machine learning models. It analyzes environmental factors such as Humidity, Temperature, and Step Count to predict stress levels.

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of various machine learning models
- Model evaluation using R² scores and visualizations
- Stress level prediction based on environmental factors

## Dataset
- The dataset used is `Stress-Lysis.csv`.
- It contains features including Humidity, Temperature, Step Count, and Stress Level.
- Missing values are handled, and categorical variables are encoded.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib & Seaborn (for visualization)
- Scikit-learn (for machine learning models)

## Machine Learning Models Implemented
- **Logistic Regression**
- **Linear Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Naïve Bayes**
- **K-Means Clustering**

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd stress-detection
   ```
3. Run the Python script:
   ```bash
   python stress_detection.py
   ```

## Results
- Models are evaluated using R² scores.
- Predictions are visualized to assess model performance.

## Future Enhancements
- Improve model accuracy with hyperparameter tuning.
- Integrate deep learning models.
- Deploy the model as a web application.

## Contributing
Feel free to contribute by creating a pull request or reporting issues.
