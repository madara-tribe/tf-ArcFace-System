import numpy as np
from sklearn.decomposition import PCA

def pca_(embedding, dim):
    assert dim < len(embedding), "dim must be less than embedding batch size"
    pca = PCA(n_components = dim)
    pca.fit(embedding)

    pca_embedding = pca.transform(embedding)
    print('pca embedding shape {}'.format(pca_embedding.shape))
    E = pca.explained_variance_ratio_
    print('left embedding contribution rate is {}'.format(np.cumsum(E)[::-1][0]))
    return pca_embedding
