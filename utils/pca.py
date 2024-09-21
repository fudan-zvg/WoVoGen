import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def downsample_clip_with_PCA(D, target=64):
    pca = PCA(n_components=target)
    reduced_features = pca.fit_transform(D)
    reduced_features = np.mean(reduced_features, axis=0)
    return reduced_features
    
def downsample_clip_with_TSNE(D, target=64):
    tsne = TSNE(n_components=target, random_state=0, method='exact')
    embedded_data = tsne.fit_transform(D)
    return embedded_data
    
if __name__ == '__main__':
    D = np.load('../output/clip_openclip/hdmap/lane.npy')
    #D = downsample_clip_with_PCA(D)
    D = downsample_clip_with_TSNE(D, 128)
    print(D.shape)