import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to visualize actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred, cluster_type, cluster_label):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Premium Amount")
    plt.ylabel("Predicted Premium Amount")
    plt.title(f"{cluster_type} - Cluster {cluster_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load dataset
df = pd.read_csv("Preprocessing_Insurance_Data_with_clusters.csv")

# Set target cluster columns and features
cluster_targets = ['Cluster_Personal', 'Cluster_Health', 'Cluster_Credit']
X_features = ['Personal_Score', 'Health_Score', 'Other_Score']
y_target = 'Premium Amount'

# Store evaluation results
results = []

# Loop over each cluster type and perform regression
for cluster_col in cluster_targets:
    print(f"\n=== Regression based on {cluster_col} ===")
    for label in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == label]

        X = cluster_df[X_features]
        y = cluster_df[y_target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Visualization
        plot_actual_vs_predicted(y_test, y_pred, cluster_col, label)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  Cluster {label}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.2f}")

        results.append({
            'Cluster Type': cluster_col,
            'Cluster Label': label,
            'MAE': mae,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        })

# Format and print final results
results_df = pd.DataFrame(results)
results_df = results_df[['Cluster Type', 'Cluster Label', 'MAE', 'RMSE', 'MSE', 'R2']]

print("\n=== Final Regression Performance Summary ===")
print(results_df.to_string(index=False, float_format="%.2f"))

# Save results to CSV
results_df.to_csv("cluster_regression_results.csv", index=False)
