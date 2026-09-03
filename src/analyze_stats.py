import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

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

def analyze_directory(directory, num_frames=20):
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

# Main execution
if __name__ == "__main__":
    base_directory = "./datasets"
    directory_names = ["1", "2", "3", "4", "5"]
    directories = [os.path.join(base_directory, name, "npy") for name in directory_names]

    all_directory_stats = []
    for directory in directories:
        print(f"Processing directory: {directory}")
        stats = analyze_directory(directory, num_frames=1000)
        all_directory_stats.append(stats)

    plot_comparison(all_directory_stats, directory_names)
