import numpy as np

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(q, h):
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1)
    h_norm = np.linalg.norm(h, axis=1)
    cosine_sim = np.dot(q, h.T)/(q_norm*h_norm+1e-10)
    return cosine_sim
