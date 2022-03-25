import numpy as np
from tqdm import tqdm


def evaluate_scheme(lsh_instance, dataset, num_query=-1, verbose=False):
    '''
    Compute the precision and recall of the specified lsh scheme on the provided dataset, taken as the average of the 
    precision and recall on each query point. If specified, will randomly select a subset of the query points to evaluate on
    '''

    # Hash the train set
    for idx, data_point in tqdm(enumerate(dataset.train_set), desc="Hashing train set", total=len(dataset.train_set)):
        lsh_instance.hash(data_point)

    # Determine the set of query points
    total_num_query_points = len(dataset.test_set)
    if num_query == -1:
        query_idxs = np.arange(total_num_query_points)
    else:
        query_idxs = np.random.permutation(total_num_query_points)[:num_query]

    # Perform the queries, and evaluate the precision / recall for each
    recalls, precisions = [], []
    for query_idx in tqdm(query_idxs, desc="Running queries", total=len(query_idxs)):
        predictions = set(lsh_instance.query(dataset.test_set[query_idx]))
        neighbors = set(dataset.neighbor_idxs[query_idx])

        intersection = predictions.intersection(neighbors)
        recall = len(intersection) / len(neighbors)
        precision = len(intersection) / len(predictions)

        recalls.append(recall)
        precisions.append(precision)

    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)

    if verbose: print(f"Avg recall: {'%.3f' % avg_recall}\nAvg precision: {'%.3f' % avg_precision}")

    return avg_recall, avg_precision

if __name__ == "__main__":
    from datasets import ANNBenchmarkDataset
    from lsh import VanillaLSH

    dataset = ANNBenchmarkDataset("sift", normalize=False)
    lsh = VanillaLSH(k=1, l=10, r=100, n_dims=dataset.dimension)

    evaluate_scheme(lsh, dataset, num_query=1000, verbose=True)