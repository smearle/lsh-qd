import os
import h5py
import wget
import argparse
import numpy as np

from tqdm import tqdm

ANN_BENCHMARKS = {"deep-image": {"dim": [96],
                                 "dist": "angular"},

                  "fashion-mnist": {"dim": [784],
                                    "dist": "euclidean"},

                  "gist": {"dim": [960],
                           "dist": "euclidean"},

                  "glove": {"dim": [25, 50, 100],
                            "dist": "euclidean"},

                  "mnist": {"dim": [784],
                            "dist": "euclidean"},

                  "nytimes": {"dim": [256],
                              "dist": "angular"},

                  "sift": {"dim": [128],
                           "dist": "euclidean"},

                  "lastfm": {"dim": [64], # on the github, this is listed as dim = 65, angular, but the filename has 64 and dot
                             "dist": "dot"}
                 }

class ANNBenchmarkDataset():
    '''
    Dataset wrapper for the approximate-nearest-neighbors benchmarks available at https://github.com/erikbern/ann-benchmarks. Collects
    a set of train points (those that will be hashed and searched over) and a set of test points (those that will be queried on), along
    with the corresponding sets of true nearest neighbors for each query and the distance to those neighbors

    Args:
        dataset_name: the name of the benchmark dataset to use, taken from the keys of ANN_BENCHMARKS
        dimension: the number of dimensions for the dataset. Most datasets have only one option (in which case dimension should be left as None),
            but for glove the user can specify one of 25, 50, or 100
        data_dir: the path to the directory where we should save / load the dataset
        normalize: whether to rescale all of the entries of each vector in the dataset to be between 0 and 1
    '''
    def __init__(self,
                 dataset_name,
                 dimension=None,
                 data_dir="./data",
                 normalize=True):

        assert dataset_name in ANN_BENCHMARKS.keys(), f"Valid model names are {list(ANN_BENCHMARKS.keys())}"

        if dimension is None:
            dimension = ANN_BENCHMARKS[dataset_name]["dim"][0]
        else:
            assert dimension in ANN_BENCHMARKS[dataset_name]["dim"], f"Valid dimensions for {dataset_name} are {ANN_BENCHMARKS[dataset_name]['dim']}"

        distance_measure = ANN_BENCHMARKS[dataset_name]["dist"]
        filename = f"{dataset_name}-{dimension}-{distance_measure}.hdf5"

        data_path = os.path.join(data_dir, filename)
        if not os.path.exists(data_path):
            download_url = f"http://ann-benchmarks.com/{filename}"
            print(f"Downloading dataset from {download_url} to {data_dir}")
            wget.download(download_url, out=data_dir)
            print("")

        data_file = h5py.File(data_path, "r")

        self.train_set, self.test_set = data_file["train"], data_file["test"]
        self.neighbor_idxs, self.neighbor_dists = data_file["neighbors"], data_file["distances"]
        self.dimension, self.distance_measure = dimension, distance_measure

        if normalize:
            min_val = min(np.min(self.train_set), np.min(self.test_set))
            max_val = max(np.max(self.train_set), np.max(self.test_set))

            self.train_set = (self.train_set - min_val) / (max_val - min_val)
            self.test_set = (self.test_set - min_val) / (max_val - min_val)

            # TODO: if we wind up needing the neighbor distances, we'll have to recompute them after we normalize the data. In that case,
            # maybe it's worth it for us to store the normalized dataset in a separate file? (Depending on how long the computation takes)


class SyntheticDataset():
    '''
    A dataset of synthetic data, where neighbors are artificially placed within a radius of each query. Both train and query points are
    uniformally distributed within the ball of specified radius

    Args:
        num_dims: the dimension of each data point
        train_size: the ultimate size of the training set (including the artificially placed neighbors)
        test_size: the size of the test set
        neighbors_per_query: the number of artificially placed neighbors for each query
        max_neighbor_dist: the radius of the ball around each query point where neighbors are placed
        radius: the radius of the ball within in which all train / query points are placed
    '''
    def __init__(self,
                 num_dims,
                 train_size,
                 test_size,
                 neighbors_per_query,
                 max_neighbor_dist,
                 radius=1):

        num_train = train_size - (test_size * neighbors_per_query)
        assert num_train >= 0, "Size of train set must exceed number of artificially placed neighbors"

        train_data = np.random.normal(0, 1, (num_train, num_dims))
        train_data = radius * (train_data / np.linalg.norm(train_data, axis=1, keepdims=True))
        
        test_data = np.random.normal(0, 1, (test_size, num_dims))
        test_data = radius * (test_data / np.linalg.norm(test_data, axis=1, keepdims=True))

        neighbor_offsets = np.random.normal(0, 1, (neighbors_per_query * test_size, num_dims))
        neighbor_offsets = max_neighbor_dist * (neighbor_offsets / np.linalg.norm(neighbor_offsets, axis=1, keepdims=True))

        neighbors = np.repeat(test_data, neighbors_per_query, axis=0) + neighbor_offsets

        self.train_set = np.concatenate([train_data, neighbors])
        self.neighbor_idxs = np.empty((test_size, neighbors_per_query))

        for query_idx in range(test_size):
            query_neighbor_idxs = range(num_train + (query_idx * neighbors_per_query), 
                                        num_train + (query_idx * neighbors_per_query) + neighbors_per_query)
            self.neighbor_idxs[query_idx, :] = query_neighbor_idxs

        self.neighbor_idxs = self.neighbor_idxs.astype(np.int32)
        self.test_set = test_data


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument()

    # d = ANNBenchmarkDataset("fashion-mnist", normalize=True)
    d = SyntheticDataset(100, 50000, 1000, 10, 0.01)
    # print(d.train_set[0])

    query_idx = 100
    query_point = d.test_set[query_idx]

    dists = []
    for neighbor_idx in d.neighbor_idxs[query_idx]:
        neighbor = d.train_set[neighbor_idx]
        dist = np.linalg.norm(query_point - neighbor)
        dists.append(dist)

    # for point in d.train_set:
    #     dist = np.linalg.norm(query_point - point)
    #     if dist < 1:
    #         print(dist)

    print("Average distance:", np.mean(dists))