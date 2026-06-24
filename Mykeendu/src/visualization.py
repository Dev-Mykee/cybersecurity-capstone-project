import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column, title=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title or f'Distribution of {column}')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
