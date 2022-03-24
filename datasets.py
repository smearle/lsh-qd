import os
import h5py
import wget
import argparse

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
    '''
    def __init__(self,
                 dataset_name,
                 dimension=None,
                 data_dir="./data"):

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


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument()

    d = ANNBenchmarkDataset("fashion-mnist")
