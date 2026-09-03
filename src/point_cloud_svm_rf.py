import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

def load_point_cloud(file_path):
    return np.load(file_path)

def preprocess_point_cloud(points):
    centroid = np.mean(points, axis=0)
    return points - centroid

def compute_basic_statistics(points):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    return mean, std

def compute_point_cloud_density(points, radius=0.1):
    tree = cKDTree(points)
    densities = tree.query_ball_point(points, r=radius, return_length=True)
    return np.mean(densities), np.std(densities)

def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.explained_variance_ratio_

def compute_local_surface_features(points, k=10):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)
    
    eigenvalues = []
    for neighbors in indices:
        cov = np.cov(points[neighbors].T)
        eig_vals = np.linalg.eigvals(cov)
        eigenvalues.append(np.sort(eig_vals)[::-1])
    
    eigenvalues = np.array(eigenvalues)
    
    linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / eigenvalues[:, 0]
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / eigenvalues[:, 0]
    sphericity = eigenvalues[:, 2] / eigenvalues[:, 0]
    
    return np.mean(linearity), np.mean(planarity), np.mean(sphericity)

def analyze_frame(points):
    basic_stats = compute_basic_statistics(points)
    density_stats = compute_point_cloud_density(points)
    pca_results = compute_pca(points)
    local_features = compute_local_surface_features(points)
    
    return {
        'basic_stats': basic_stats,
        'density_stats': density_stats,
        'pca_results': pca_results,
        'local_features': local_features
    }

def analyze_directory(directory, num_frames=1000):
    all_stats = []
    for i in range(num_frames):
        file_name = f"{i:06d}_objects.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        points = load_point_cloud(file_path)
        points = preprocess_point_cloud(points)
        frame_stats = analyze_frame(points)
        all_stats.append(frame_stats)
    return all_stats

def plot_comparison(all_directory_stats, directory_names):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)

    # Plot mean positions
    ax_mean = fig.add_subplot(gs[0, :])
    for dir_stats, dir_name in zip(all_directory_stats, directory_names):
        for i, dim in enumerate(['X', 'Y', 'Z']):
            means = [stat['basic_stats'][0][i] for stat in dir_stats]
            ax_mean.plot(means, label=f'{dir_name} {dim}')
    ax_mean.set_title('Mean Positions')
    ax_mean.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_mean.set_xlabel('Frame')
    ax_mean.set_ylabel('Position')

    # Plot densities
    ax_density = fig.add_subplot(gs[1, 0])
    for dir_stats, dir_name in zip(all_directory_stats, directory_names):
        densities = [stat['density_stats'][0] for stat in dir_stats]
        ax_density.plot(densities, label=dir_name)
    ax_density.set_title('Point Cloud Densities')
    ax_density.legend()
    ax_density.set_xlabel('Frame')
    ax_density.set_ylabel('Density')

    # Plot PCA results
    ax_pca = fig.add_subplot(gs[1, 1])
    x = np.arange(3)
    width = 0.15
    for i, (dir_stats, dir_name) in enumerate(zip(all_directory_stats, directory_names)):
        pca_results = np.mean([stat['pca_results'] for stat in dir_stats], axis=0)
        ax_pca.bar(x + i*width, pca_results, width, label=dir_name)
    ax_pca.set_title('PCA Explained Variance Ratios')
    ax_pca.set_xticks(x + width * 2)
    ax_pca.set_xticklabels(['PC1', 'PC2', 'PC3'])
    ax_pca.legend()
    ax_pca.set_ylabel('Explained Variance Ratio')

    # Plot local surface features
    features = ['Linearity', 'Planarity', 'Sphericity']
    for i, feature in enumerate(features):
        ax = fig.add_subplot(gs[2, i])
        for dir_stats, dir_name in zip(all_directory_stats, directory_names):
            values = [stat['local_features'][i] for stat in dir_stats]
            ax.plot(values, label=dir_name)
        ax.set_title(f'Local Surface Feature: {feature}')
        ax.legend()
        ax.set_xlabel('Frame')
        ax.set_ylabel(feature)

    plt.tight_layout()
    plt.show()

def plot_autocorrelation(all_directory_stats, directory_names):
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    features = ['Density', 'Linearity', 'Planarity', 'Sphericity', 'PC1']
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        for dir_stats, dir_name in zip(all_directory_stats, directory_names):
            if feature == 'Density':
                values = [stat['density_stats'][0] for stat in dir_stats]
            elif feature == 'PC1':
                values = [stat['pca_results'][0] for stat in dir_stats]
            else:
                index = ['Linearity', 'Planarity', 'Sphericity'].index(feature)
                values = [stat['local_features'][index] for stat in dir_stats]
            
            plot_acf(values, ax=ax, lags=50, alpha=0.05, label=dir_name)
        
        ax.set_title(f'Autocorrelation: {feature}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_ylim(-1, 1)  # Set y-axis range from -1 to 1
    
    # Remove the extra subplot at the bottom
    axes[-1, -1].remove()
    
    # Add a shared legend at the bottom of the plot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(directory_names), bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Leave space for the legend
    plt.show()

def plot_feature_distributions(all_directory_stats, directory_names):
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    features = ['Density', 'Linearity', 'Planarity', 'Sphericity', 'PC1']
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        for dir_stats, dir_name in zip(all_directory_stats, directory_names):
            if feature == 'Density':
                values = [stat['density_stats'][0] for stat in dir_stats]
            elif feature == 'PC1':
                values = [stat['pca_results'][0] for stat in dir_stats]
            else:
                index = ['Linearity', 'Planarity', 'Sphericity'].index(feature)
                values = [stat['local_features'][index] for stat in dir_stats]
            
            sns.kdeplot(values, ax=ax, label=dir_name, shade=True)
        
        ax.set_title(f'Distribution: {feature}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def extract_time_series_features(feature_sequence):
    return np.array([
        np.mean(feature_sequence),
        np.std(feature_sequence),
        skew(feature_sequence),
        kurtosis(feature_sequence),
        np.max(feature_sequence) - np.min(feature_sequence)  # Range
    ])

def prepare_features(all_directory_stats):
    all_features = []
    for dir_stats in all_directory_stats:
        for frame_stat in dir_stats:
            frame_features = []
            # Density features
            frame_features.extend(frame_stat['density_stats'])
            # Local surface features
            frame_features.extend(frame_stat['local_features'])
            # PCA results
            frame_features.extend(frame_stat['pca_results'])
            all_features.append(frame_features)
    return np.array(all_features)

def train_and_evaluate_model(X, y, model_type='svm'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'svm':
        model = SVC(kernel='rbf', C=1, gamma='scale')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"{model_type.upper()} Model Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def visualize_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    base_directory = "./datasets"
    directory_names = ["1", "2", "3", "4", "5"]
    directories = [os.path.join(base_directory, name, "npy") for name in directory_names]

    all_directory_stats = []
    for directory in directories:
        print(f"Processing directory: {directory}")
        stats = analyze_directory(directory, num_frames=1000)  # Analyze all 1000 frames
        all_directory_stats.append(stats)

    # Prepare features and labels
    all_features = prepare_features(all_directory_stats)
    labels = np.repeat(np.arange(len(all_directory_stats)), [len(stats) for stats in all_directory_stats])

    # Define feature names (for visualization)
    feature_names = ['Density_Mean', 'Density_Std', 'Linearity', 'Planarity', 'Sphericity', 'PC1', 'PC2', 'PC3']

    # Train and evaluate SVM model
    svm_model, svm_scaler = train_and_evaluate_model(all_features, labels, model_type='svm')

    # Train and evaluate Random Forest model
    rf_model, rf_scaler = train_and_evaluate_model(all_features, labels, model_type='rf')

    # Visualize feature importance for Random Forest
    visualize_feature_importance(rf_model, feature_names)

