# Machine Learning Regression Models

This project compares multiple machine learning regression models to predict outputs based on input data. The models included in this project are:

- Linear Regression
- Polynomial Regression (Degree 2)
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)

## Project Overview

The goal of this project is to evaluate and compare different regression models on a given dataset to predict an output variable (`y`) based on two input variables (`input1` and `input2`). The models are evaluated based on three performance metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn

You can install them using the following command:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset used in this project is a `.xlsx` file (`data 1 1.xlsx`) containing two input columns (`input1`, `input2`) and one output column (`output`). The dataset is loaded and preprocessed to split into training and testing sets for model evaluation.

### Example Dataset Format:

| input1 | input2 | output |
|--------|--------|--------|
|   ...  |   ...  |   ...  |
|   ...  |   ...  |   ...  |

## Code Overview

### Steps:

1. **Load Dataset**: The dataset is loaded from an Excel file.
2. **Data Splitting**: The data is split into training and testing sets using an 80-20 split ratio.
3. **Model Training**: A dictionary of models is defined and trained on the training data.
4. **Model Evaluation**: Each model is evaluated on the test data using MAE, MSE, and R² Score.
5. **Results**: The results are collected in a DataFrame and displayed for comparison.

### Code Execution:

To run the script, simply run the Python code, and the results will be printed as a comparison of each model's performance:

```bash
python main.py
```

### Example Output:

```
                        MAE       MSE   R2 Score
Linear Regression        0.1234    0.0456   0.89
Polynomial Regression    0.1123    0.0394   0.91
Ridge Regression         0.1189    0.0421   0.90
Lasso Regression         0.1245    0.0471   0.88
Decision Tree            0.1452    0.0563   0.82
Random Forest            0.1156    0.0410   0.92
Gradient Boosting        0.1084    0.0372   0.93
SVR                      0.1323    0.0504   0.87
K-Nearest Neighbors      0.1376    0.0531   0.85
```

## Conclusion

This project demonstrates how to compare multiple regression models on a given dataset, helping to choose the best model for a specific task based on their performance metrics.

Feel free to modify the dataset and experiment with different hyperparameters to further improve the results.

## License
MIT License

## Author
andrepradika
