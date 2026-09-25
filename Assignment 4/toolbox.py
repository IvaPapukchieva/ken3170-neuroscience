import numpy as np

def compute_rdm(patterns: np.ndarray) -> np.ndarray:
    """Compute the Euclidean-distance RDM for a set of representational patterns."""
    n_classes = patterns.shape[0]
    rdm = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            distance = np.linalg.norm(patterns[i] - patterns[j])
            rdm[i][j]= distance
            rdm[j][i]= distance
    
    return rdm


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Extract the off-diagonal upper-triangular entries of an RDM as a 1D vector."""
    n_classes = rdm.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2
    vector = np.zeros(n_pairs)
    index = 0
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            vector[index] = rdm[i][j]
            index+=1
    return vector
       


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Compare two RDMs via Pearson correlation of their upper-triangular entries."""
    vector_a = upper_triangle(rdm_a)
    vector_b = upper_triangle(rdm_b)
    # means
    mean_a = np.mean(vector_a)
    mean_b = np.mean(vector_b)

    # differences
    diff_a = vector_a - mean_a
    diff_b = vector_b - mean_b

    # numerator
    numerator = np.sum(diff_a * diff_b)
    # denominator
    denominator = np.sqrt(np.sum(diff_a ** 2) * np.sum(diff_b ** 2))
    # pearson correlation
    if denominator != 0:
        correlation = numerator / denominator
    else:
        raise ValueError(
        "Pearson correlation is undefined: one or both RDM vectors have zero variance."
    )
    return correlation