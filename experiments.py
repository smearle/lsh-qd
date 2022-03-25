import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool


def evaluate_scheme(lsh_instance, dataset, num_query=-1, verbose=False):
    '''
    Compute the precision and recall of the specified lsh scheme on the provided dataset, taken as the average of the 
    precision and recall on each query point. If specified, will randomly select a subset of the query points to evaluate on
    '''

    # Hash the train set
    for idx, data_point in tqdm(enumerate(dataset.train_set), desc="Hashing train set", total=len(dataset.train_set)):
        lsh_instance.new_hash(data_point)

    # print("Hashing train set")
    # with Pool(4) as pool:
        # tqdm(pool.imap(lsh_instance.new_hash, list(dataset.train_set)), desc="Hashing train set", total=len(dataset.train_set))
        # pool.map(lsh_instance.new_hash, dataset.train_set)

    # Determine the set of query points
    total_num_query_points = len(dataset.test_set)
    if num_query == -1:
        query_idxs = np.arange(total_num_query_points)
    else:
        query_idxs = np.random.permutation(total_num_query_points)[:num_query]

    # Perform the queries, and evaluate the precision / recall for each
    recalls, precisions, num_predictions = [], [], []
    for query_idx in tqdm(query_idxs, desc="Running queries", total=len(query_idxs)):
        # if isinstance(lsh_instance, MultiProbeLSH): lsh_instance.query([], num_probes)
        # elif isinstance(lsh_instance, AlphaLSH): 
        predictions = set(lsh_instance.query(dataset.test_set[query_idx]))
        neighbors = set(dataset.neighbor_idxs[query_idx])

        intersection = predictions.intersection(neighbors)
        recall = len(intersection) / len(neighbors)

        if len(predictions) != 0:
            precision = len(intersection) / len(predictions)
            precisions.append(precision)

        recalls.append(recall)
        num_predictions.append(len(predictions))

    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    avg_num_predictions = np.mean(num_predictions)

    if verbose: print(f"Avg num predicted: {'%.3f' % avg_num_predictions}\nAvg recall: {'%.3f' % avg_recall}\nAvg precision: {'%.3f' % avg_precision}")

    return avg_recall, avg_precision

def calculate_average_projected_distance(dataset, num_projections=100, num_query=-1):
    A = np.random.normal(0, 1, size=(num_projections, dataset.dimension)) # (num_projections, n_dims)

    # Determine the set of query points
    total_num_query_points = len(dataset.test_set)
    if num_query == -1:
        query_idxs = np.arange(total_num_query_points)
    else:
        query_idxs = np.random.permutation(total_num_query_points)[:num_query]

    all_distances = []
    for query_idx in tqdm(query_idxs, desc="Computing distance from query points to neighbors", total=len(query_idxs)):
        projected = np.matmul(A, dataset.test_set[query_idx]) # (num_projections,)

        neighbors = np.array([dataset.train_set[idx] for idx in dataset.neighbor_idxs[query_idx]]) # (100, n_dims)

        neighbor_projections = np.matmul(A, neighbors.T) # (num_projections, 100)

        distances = np.abs(neighbor_projections - projected)
        all_distances += distances.flatten().tolist()
    
    avg_distance = np.mean(all_distances)
    print("Average projected distance from query point to neighbor:", avg_distance)

if __name__ == "__main__":
    from datasets import ANNBenchmarkDataset
    from lsh import VanillaLSH

    dataset = ANNBenchmarkDataset("fashion-mnist", normalize=False)
    lsh = VanillaLSH(k=55, l=100, r=15000, n_dims=dataset.dimension)

    evaluate_scheme(lsh, dataset, num_query=1000, verbose=True)
    # calculate_average_projected_distance(dataset, num_query=1000)