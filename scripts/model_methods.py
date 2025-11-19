import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(data_path):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    df = df.astype(float)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df

def vis_heatmap(df):
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
    return KNeighborsRegressor(k_value)
