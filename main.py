import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loading import load_images_from_folder
from src.coarse_recognition import estimate_gaussian_params, likelihood_ratio_test
from src.fine_recognition import compute_wasserstein_distance, adaptive_weighting
from src.fusion_criterion import final_fusion_criterion
from src.feature_extraction import extract_features

def calculate_similarity_score(feature1, feature2):
    feature1 = np.array(feature1).reshape(1, -1)
    feature2 = np.array(feature2).reshape(1, -1)
    return cosine_similarity(feature1, feature2)[0][0]

def generate_genuine_imposter_scores(gallery_features, probe_features, labels):
    genuine_scores = []
    imposter_scores = []

    for i, feat_gallery in enumerate(gallery_features):
        for j, feat_probe in enumerate(probe_features):
            score = calculate_similarity_score(feat_gallery, feat_probe)
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else: 
                imposter_scores.append(score)

    return np.array(genuine_scores), np.array(imposter_scores)

def main():
    gallery_folder = 'data/session1'
    probe_folder = 'data/session2'

    gallery_images = load_images_from_folder(gallery_folder)
    probe_images = load_images_from_folder(probe_folder)

    sigma = 4.6
    delta = 2.6
    omega = 0.58
    kappa = 2.65
    num_gabor_filters = 6 

    frequencies = [omega] * num_gabor_filters
    thetas = [i * (np.pi / num_gabor_filters) for i in range(num_gabor_filters)]

    gallery_features = extract_features(gallery_images, frequencies, thetas, sigma, delta, kappa)
    probe_features = extract_features(probe_images, frequencies, thetas, sigma, delta, kappa)

    labels = [int(filename[:5]) // 10 for filename in sorted(os.listdir(gallery_folder))]

    genuine_scores, imposter_scores = generate_genuine_imposter_scores(gallery_features, probe_features, labels)
    print("Genuine scores shape:", genuine_scores.shape)
    print("Imposter scores shape:", imposter_scores.shape)
    alpha = 1e-5
    beta = 5e-10
    gamma_alpha = 230
    gamma_beta = 5e28
    gen_params = estimate_gaussian_params(genuine_scores)
    imp_params = estimate_gaussian_params(imposter_scores)
    if genuine_scores.size > 0 and imposter_scores.size > 0:
        w = compute_wasserstein_distance(genuine_scores, imposter_scores)
    else:
        print("Warning: Genuine or imposter scores are empty.")
        return

    z = 2
    tau = adaptive_weighting(w, w, z=z)
    predictions = []
    for probe in probe_features:
        L_k = likelihood_ratio_test(probe, gen_params, imp_params)
        x_k = gamma_alpha*float(np.mean(probe)) 
        fusion_score = final_fusion_criterion(L_k, x_k, tau, gamma_alpha, gamma_beta, i0=1.0, a1=0.5)

        prediction = 'genuine' if np.mean(fusion_score) > 0.5 else 'imposter'
        predictions.append(prediction)

    genuine_count = predictions.count('genuine')
    total_count = len(predictions)
    percentage = (genuine_count / total_count) * 100
    print(f"Percentage of genuine images: {percentage:.2f}%")

if __name__ == "__main__":
    main()
