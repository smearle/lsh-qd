import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool\

from lsh import VanillaLSH, AlphaLSH, MultiProbeLSH, AlphaMultiProbeLSH
from datasets import ANNBenchmarkDataset, SyntheticDataset


def evaluate_scheme(lsh_instance, dataset, num_query=-1, verbose=False, **kwargs):
    '''
    Compute the precision and recall of the specified lsh scheme on the provided dataset, taken as the average of the 
    precision and recall on each query point. If specified, will randomly select a subset of the query points to evaluate on
    '''

    # Hash the train set
    for idx, data_point in tqdm(enumerate(dataset.train_set), desc="Hashing train set", total=len(dataset.train_set)):
        lsh_instance.hash(data_point)

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
        predictions = set(lsh_instance.query(dataset.test_set[query_idx], **kwargs))
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


def report_query(lsh):
    avg_overall_time = np.mean(lsh.query_total_times)
    avg_hash_time = np.mean(lsh.query_hash_times)
    avg_set_const_time = np.mean(lsh.query_set_construction_times)
    avg_scan_time = np.mean(lsh.query_scan_times)
    avg_pert_collisions = np.mean(lsh.query_num_perturbed_bucket_collisions)
    avg_orig_collisions = np.mean(lsh.query_num_original_bucket_collisions)

    print("=" * 100)
    print(f"Query Report for {type(lsh)} Type LSH (l = {lsh.l}, k = {lsh.k})")
    print(f"Average overall query time: {avg_overall_time}")
    print(f"Average hash time: {'%.4e' % avg_hash_time} ({'%.2f' % (100 * avg_hash_time / avg_overall_time)}%)")
    print(f"Average set construction time: {'%.4e' % avg_set_const_time} ({'%.2f' % (100 * avg_set_const_time / avg_overall_time)}%)")
    print(f"Average scan time: {'%.4e' % avg_scan_time} ({'%.2f' % (100 * avg_scan_time / avg_overall_time)}%)")
    print(f"Average number of collisions in perturbed buckets: {'%.2f' % avg_pert_collisions}")
    print(f"Average number of collisions in original buckets: {'%.2f' % avg_orig_collisions}")
    print("=" * 100)



if __name__ == "__main__":
    

    # dataset = ANNBenchmarkDataset("fashion-mnist", normalize=False)
    # calculate_average_projected_distance(dataset, num_query=1000)

    # lsh = MultiProbeLSH(k=20, l=20, r=500, n_dims=dataset.dimension)


    # lsh = VanillaLSH(k=65, l=100, r=15000, n_dims=dataset.dimension)
    # lsh = AlphaLSH(k=25, l=150, r=15000, n_dims=dataset.dimension)
    # lsh = MultiProbeLSH(k=75, l=50, r=10000, n_dims=dataset.dimension)
    # lsh = AlphaMultiProbeLSH(k=30, l=40, r=10000, n_dims=dataset.dimension)
    # evaluate_scheme(lsh, dataset, num_query=100, verbose=True, num_perturbations=25, alpha=5)


    
    # lsh = VanillaLSH(k=12, l=10, r=1, n_dims=dataset.dimension)
    # lsh = AlphaLSH(k=1, l=15, r=1, n_dims=dataset.dimension)
    

    # evaluate_scheme(lsh, dataset, num_query=1000, verbose=True, timed=True)
    # evaluate_scheme(lsh, dataset, num_query=1000, verbose=True, alpha=13, timed=True)
    

    

    # calculate_average_projected_distance(dataset, num_query=1000)


    dataset = SyntheticDataset(num_dims=100, train_size=50000, test_size=1000, neighbors_per_query=10, max_neighbor_dist=0.01)
    multi_lsh = MultiProbeLSH(k=12, l=3, r=1, n_dims=dataset.dimension)
    alpha_multi_lsh = AlphaMultiProbeLSH(k=3, l=3, r=1, n_dims=dataset.dimension)

    evaluate_scheme(multi_lsh, dataset, num_query=500, verbose=True, num_perturbations=2, timed=True)
    report_query(multi_lsh)
    
    evaluate_scheme(alpha_multi_lsh, dataset, num_query=500, verbose=True, num_perturbations=2, alpha=3, timed=True)
    report_query(alpha_multi_lsh)