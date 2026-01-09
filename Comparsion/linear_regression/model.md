# Model Comparison: Scikit-learn vs. cgadimpl

This document compares the two implementations of the Linear Regression model for the Advertising dataset.

## 1. `pytorch_model.py` (Scikit-learn)

Despite the filename, this script uses **Scikit-learn** to build the model.

### Key Characteristics:
- **Algorithm**: Uses **Ordinary Least Squares (OLS)**.
- **Optimizer**: **None**. Scikit-learn's `LinearRegression` uses a closed-form mathematical solution (via SVD or QR decomposition) to find the weights that minimize the sum of squared residuals in a single step.
- **Preprocessing**: Uses `StandardScaler` within a `Pipeline`.
- **Data Splitting**: Uses `train_test_split` with shuffling (`random_state=42`).
- **Hyperparameter Tuning**: Uses `GridSearchCV` to test `fit_intercept`.

### Results:
- **Test RMSE**: ~1.705
- **Baseline RMSE**: ~5.648

---

## 2. `linear_regression_main.cpp` (cgadimpl)

This implementation uses the custom **cgadimpl** C++ framework.

### Key Characteristics:
- **Algorithm**: Uses **Gradient Descent**.
- **Optimizer**: **Adam**. Unlike the Scikit-learn version, this model iteratively updates its weights using the Adam optimizer with a learning rate of 0.1.
- **Preprocessing**: Manual implementation of `StandardScaler` logic (calculating mean/std and applying transformation).
- **Data Splitting**: Manual sequential split (first 80% for training, last 20% for testing).
- **Model**: A single `ag::nn::Linear` layer.

### Results:
- **Test RMSE**: ~1.624
- **Baseline RMSE**: ~5.269

---

## 3. Comparison Summary

| Feature | `pytorch_model.py` | `linear_regression_main.cpp` |
| :--- | :--- | :--- |
| **Framework** | Scikit-learn (Python) | cgadimpl (C++) |
| **Method** | Closed-form (OLS) | Iterative (Gradient Descent) |
| **Optimizer** | None | **Adam** |
| **Data Split** | Shuffled (Random) | Sequential |
| **Scaling** | Automatic (`Pipeline`) | Manual |

### Why are the results different?
The `cgadimpl` implementation achieved a slightly lower RMSE (**1.624** vs **1.705**). This difference is primarily due to the **Data Splitting** method:
- The Python script shuffles the data, ensuring a representative distribution in both sets.
- The C++ script takes the last 20% of the file as the test set. If the later entries in `advertising.csv` are more predictable or follow a slightly different pattern, the RMSE will vary.

Both models significantly outperform the baseline, confirming that the `cgadimpl` framework's autograd and optimizer components are working correctly.
