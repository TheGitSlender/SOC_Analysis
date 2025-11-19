import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_path):
    """Load and standardize the dataset"""
    
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    df = df.astype(float)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df

def vis_heatmap(df):
    """
    Create a correlation heatmap for the dataset
    """

    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                fmt='.2f',
                ax=ax)
    ax.set_title('Correlation Matrix')
    return fig

def model(k_value):
    """
    Create a KNN Regressor model
    """
    return KNeighborsRegressor(k_value)

def train_and_evaluate_model(data, selected_features, k_value, cv_folds=5):
    """
    Train KNN model with cross-validation and return scores 
    """
    # Separate features and target
    X = data[selected_features]
    y = data['cpu_load']

    knn_model = KNeighborsRegressor(n_neighbors=k_value)
    cv_scores = cross_val_score(knn_model, X, y, cv=cv_folds, scoring='r2')

    knn_model.fit(X, y)

    return {
        'mean_r2': np.mean(cv_scores),
        'std_r2': np.std(cv_scores),
        'scores': cv_scores,
        'model': knn_model
    }

def predict_cpu_load(model, feature_values, selected_features):
    """
    Predict CPU load for new observation
    """

    X_new = pd.DataFrame([feature_values])[selected_features]
    prediction = model.predict(X_new)
    return prediction[0]


