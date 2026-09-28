import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def calculate_clustering_metrics(true_label, pred_label):
    true_label = np.asarray(true_label, dtype=int)
    pred_label = np.asarray(pred_label, dtype=int)

    nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
    ari = metrics.adjusted_rand_score(true_label, pred_label)
    cs = metrics.completeness_score(true_label, pred_label)

    num_classes = int(max(true_label.max(), pred_label.max()) + 1)
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (true_label, pred_label), 1)
    true_ids, pred_ids = linear_sum_assignment(-counts)
    mapping = {int(pred): int(true) for true, pred in zip(true_ids, pred_ids)}
    remapped = np.asarray([mapping.get(int(pred), int(pred)) for pred in pred_label], dtype=int)

    acc = metrics.accuracy_score(true_label, remapped)
    f1 = metrics.f1_score(true_label, remapped, average="macro")
    return acc, nmi, f1, ari, cs
