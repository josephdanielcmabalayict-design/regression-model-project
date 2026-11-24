
"""
LINEAR ALGEBRA FOR AI FINAL PROJECT
Student Academic Performance Prediction
SDG 3: Good Health and Well-being

Dataset: 150 synthetic student observations
Model: Multiple Linear Regression (OLS)
Target: Academic Score (0-100)

Dependencies:
numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURES = [
    "Sleep_Hours_Per_Night",
    "Study_Hours_Per_Day",
    "Stress_Level_1_to_10"
]

TARGET = "Academic_Score_0_to_100"

DATASET_FILE = "synth.csv"


# ============================================================
# A. DATASET LOADING AND EXPLORATION
# ============================================================

df = pd.read_csv(DATASET_FILE)

print("\n========================================")
print(" STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("========================================")

print("\nDATASET INFORMATION")
print("Number of samples:", len(df))
print("Number of independent variables:", len(FEATURES))
print("Independent variables:", FEATURES)
print("Dependent variable:", TARGET)

print("\nDataset preview:")
print(df[FEATURES + [TARGET]].head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df[FEATURES + [TARGET]].isnull().sum())

# Extract input matrix and target vector
X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

# 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

n, p = X_train.shape

print("\nTRAINING AND TESTING DATA")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", p)


# ============================================================
# B. MATRIX REPRESENTATION
# ============================================================

print("\n========== PART B: MATRICES ==========")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_train transpose shape:", X_train.T.shape)

gram_raw = X_train.T @ X_train

print("\nRaw Gram Matrix:")
print(gram_raw)

print("Gram matrix shape:", gram_raw.shape)


# ============================================================
# C. VECTOR SPACES AND MATRIX RANK
# ============================================================

print("\n========== PART C: VECTOR SPACES ==========")

rank = np.linalg.matrix_rank(X_train)

print("Feature-space dimension:", p)
print("Matrix rank:", rank)
print("Maximum possible rank:", min(X_train.shape))

if rank == p:
    print("The matrix has full column rank.")
else:
    print("The matrix does not have full column rank.")


# ============================================================
# D. STANDARDIZATION AND LINEAR TRANSFORMATION
# ============================================================

print("\n========== PART D: STANDARDIZATION ==========")

scaler = StandardScaler()

# Fit using training data only
X_train_std = scaler.fit_transform(X_train)

# Apply the same transformation to testing data
X_test_std = scaler.transform(X_test)

# Diagonal scaling matrix
M = np.diag(1.0 / scaler.scale_)

print("Training means:", scaler.mean_)
print("Training standard deviations:", scaler.scale_)

print("\nDiagonal scaling matrix:")
print(M)

print("\nStandardized training means:")
print(X_train_std.mean(axis=0))

print("Standardized training standard deviations:")
print(X_train_std.std(axis=0))

print("\nFirst raw test observation:")
print(X_test[0])

print("First standardized test observation:")
print(X_test_std[0])

print("Determinant of scaling matrix:", np.linalg.det(M))


# ============================================================
# E. DETERMINANTS AND MATRIX CONDITIONING
# ============================================================

print("\n========== PART E: DETERMINANTS ==========")

G = X_train_std.T @ X_train_std

print("Standardized Gram Matrix:")
print(G)

print("\nDeterminant of G:", np.linalg.det(G))
print("Condition number of G:", np.linalg.cond(G))
print("Rank of G:", np.linalg.matrix_rank(G))


# ============================================================
# F. COVARIANCE, EIGENVALUES AND EIGENVECTORS
# ============================================================

print("\n========== PART F: EIGENVALUES ==========")

C = np.cov(X_train_std, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(C)

# Sort eigenvalues from largest to smallest
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(variance_ratio)

print("Covariance Matrix:")
print(C)

print("\nEigenvalues:")
print(eigenvalues)

print("\nExplained Variance Ratio:")
print(variance_ratio)

print("\nCumulative Explained Variance:")
print(cumulative_variance)

# Verify an eigenpair
v1 = eigenvectors[:, 0]

residual = np.linalg.norm(
    C @ v1 - eigenvalues[0] * v1
)

print("\nEigenpair verification error:", residual)


# ============================================================
# G. DIAGONALIZATION AND PCA
# ============================================================

print("\n========== PART G: PCA ==========")

V = eigenvectors
Lambda = np.diag(eigenvalues)

# Reconstruct covariance matrix
C_reconstructed = V @ Lambda @ V.T

reconstruction_error = np.linalg.norm(
    C - C_reconstructed,
    ord="fro"
)

print("V shape:", V.shape)
print("Lambda shape:", Lambda.shape)

print("Reconstruction error:", reconstruction_error)

# Determine number of components needed for 95% variance
k = int(np.argmax(cumulative_variance >= 0.95) + 1)

V_k = V[:, :k]

# Transform training and testing data
X_train_pca = X_train_std @ eigenvectors[:, :k] # X_train_std @ V_k changed, per https://chatgpt.com/s/t_6b04d3b855bc81918262f55745c631f9
X_test_pca = X_test_std @ eigenvectors[:, :k] #X_test_std @ V_k

print("\nPrincipal components retained:", k)
print("Cumulative variance retained:", cumulative_variance[k-1])

print("PCA training shape:", X_train_pca.shape)
print("PCA testing shape:", X_test_pca.shape)


# ============================================================
# H. ORDINARY LEAST SQUARES REGRESSION
# ============================================================

print("\n========== PART H: OLS REGRESSION ==========")

def fit_ols(A, target):

    # Add intercept column
    A_bias = np.column_stack([
        np.ones(A.shape[0]), A
    ])

    # Calculate regression coefficients
    beta, *_ = np.linalg.lstsq(
        A_bias, target, rcond=None
    )

    return beta


def predict(A, beta):

    A_bias = np.column_stack([
        np.ones(A.shape[0]), A
    ])

    return A_bias @ beta


# Train full-feature model
beta_full = fit_ols(X_train_std, y_train)

# Train PCA model
beta_pca = fit_ols(X_train_pca, y_train)

# Generate predictions
pred_full = predict(X_test_std, beta_full)
pred_pca = predict(X_test_pca, beta_pca)


# ============================================================
# NORMAL EQUATION DEMONSTRATION
# ============================================================

print("\nNORMAL EQUATION DEMONSTRATION")

X_bias = np.column_stack([
    np.ones(X_train_std.shape[0]),
    X_train_std
])

XTX = X_bias.T @ X_bias
XTy = X_bias.T @ y_train

print("XTX shape:", XTX.shape)
print("XTy shape:", XTy.shape)

if np.linalg.matrix_rank(XTX) == XTX.shape[0]:

    beta_normal = np.linalg.solve(XTX, XTy)

    print("Normal equation coefficients:")
    print(beta_normal)

    print("\nOLS solver coefficients:")
    print(beta_full)

    print("\nDifference between solutions:")
    print(np.linalg.norm(beta_normal - beta_full))

else:
    print("Normal equation matrix is singular.")
    print("Using least-squares solver instead.")


# ============================================================
# REGRESSION COEFFICIENT INTERPRETATION
# ============================================================

print("\nREGRESSION COEFFICIENTS")

print("Intercept:", beta_full[0])

for feature, coefficient in zip(
    FEATURES, beta_full[1:]
):
    print(f"{feature}: {coefficient:.4f}")


# ============================================================
# I. MODEL EVALUATION
# ============================================================

print("\n========== PART I: MODEL EVALUATION ==========")

def evaluate(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return rmse, r2


rmse_full, r2_full = evaluate(y_test, pred_full)

rmse_pca, r2_pca = evaluate(y_test, pred_pca)

comparison = pd.DataFrame({
    "Model": ["Full OLS", "PCA OLS"],
    "RMSE": [rmse_full, rmse_pca],
    "R2": [r2_full, r2_pca]
})

print(comparison.to_string(index=False))

print("\nTarget mean:", np.mean(y_test))

print("\nFirst actual score:", y_test[0])
print("First predicted score:", pred_full[0])


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

print("\n========== RESIDUAL ANALYSIS ==========")

residuals = y_test - pred_full

print("Residuals:", residuals)
print("Mean residual:", np.mean(residuals))
print("Residual sum of squares:", np.sum(residuals**2))


# ============================================================
# J. VISUALIZATIONS
# ============================================================

# 1. Correlation Matrix

correlation_matrix = df[FEATURES].corr()

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(range(p), FEATURES, rotation=30, ha="right")
plt.yticks(range(p), FEATURES)

plt.title("Student Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=180)
plt.show()


# 2. PCA Scree Plot

plt.figure(figsize=(6, 4))

plt.plot(
    range(1, p + 1),
    variance_ratio,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")

plt.xticks(range(1, p + 1))
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("scree_plot.png", dpi=180)
plt.show()


# 3. PCA Projection

plt.figure(figsize=(6, 4))

points = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="viridis",
    alpha=0.85
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Student Data Projected onto PCA Components")

plt.colorbar(
    points,
    label="Academic Score"
)

plt.tight_layout()
plt.savefig("pca_projection.png", dpi=180)
plt.show()


# 4. Predicted vs Actual Academic Scores

plt.figure(figsize=(6, 5))

plt.scatter(
    y_test,
    pred_full,
    alpha=0.85
)

low = min(y_test.min(), pred_full.min())
high = max(y_test.max(), pred_full.max())

plt.plot(
    [low, high],
    [low, high],
    "r--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Academic Score")
plt.ylabel("Predicted Academic Score")
plt.title("Predicted vs Actual Academic Performance")

plt.legend()
plt.tight_layout()

plt.savefig("predicted_vs_actual.png", dpi=180)
plt.show()


# 5. Residual Plot

plt.figure(figsize=(6, 4))

plt.scatter(
    pred_full,
    residuals,
    alpha=0.85
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Academic Score")
plt.ylabel("Residual")
plt.title("Residual Analysis")

plt.tight_layout()
plt.savefig("residual_plot.png", dpi=180)
plt.show()


print("\n========================================")
print(" PROJECT EXECUTION COMPLETED")
print("========================================")


print("Number of PCA components:", k)
print("Original features:", X_train_std.shape[1])
print("PCA features:", X_train_pca.shape[1])
print("Explained variance:", cumulative_variance[k-1])
