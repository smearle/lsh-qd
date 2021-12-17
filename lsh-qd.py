import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import jaccard


class AlphaMinhash():
    '''
    An implementation of the Minhash algorithm for approximate nearest
    neighbors on binary strings, designed for use in niching and QD. In
    addition to the typical parameters of # of tables and # of bands, this
    also accepts an alpha parameter when querying. An item in the database
    is returned only if it is hashed into the same bucket as the query in
    at least alpha of the t tables.
    Args:
        t: the number of distinct hash tables
        r: the number of bands (i.e. hash functions) used to determine an item's
            location in each of the t tables
        seed: the random seed for generating the hash functions
    '''

    def __init__(self,
                 t,
                 r,
                 seed=42):
        self.t, self.r = t, r
        self.seed = seed
        # We use the largest 32-bit integer as our maximum hash value, but
        # this could be changed without too much effort
        self.max_val = 2 ** 32 - 1
        self.p = 4294967311
        # Store each of hash function in an array of size (t, r). self.hash_functions[i][j]
        # indicates the j'th band of the i'th table
        self.hash_functions = [[self._generate_hash_function() for _ in range(self.r)]
                               for _ in range(self.t)]
        # This dictionary maps from a data index to a list of t bucket ids
        # indicating the keys in each hash table where the index can be found
        self.cur_data_idx = 0
        self.data_idx_to_bucket_ids = {}
        # This list stores t dictionaries, each of which maps from a hash value /
        # bucket id to each of the data indexes which have been hashed to that bucket
        self.tables = [defaultdict(list) for _ in range(self.t)]

    def _generate_hash_function(self):
        '''
        Returns a single random hash function that maps from an integer to
        a value in the interval [0, 1]
        '''
        a = np.random.randint(0, self.max_val)
        b = a
        while b == a:
            b = np.random.randint(0, self.max_val)

        def hash_func(x):
            return ((a * x + b) % self.p) / self.p

        return hash_func

    def minhash(self, x):
        '''
        Performs the Minhash algorithm on the provided binary string, hashing the data
        into each of the t tables. Also returns the hash indices
        Args:
            x: the input data, a binary string
        '''
        nonzero_indexes = np.where(x == 1)[0]
        if len(nonzero_indexes) == 0:
            return
        all_bucket_ids = []
        # For each set of r bands...
        for hash_idx, bands in enumerate(self.hash_functions):
            # Compute the data point's signature under those hash functions, and its
            # corresponding bucket id
            signature = [np.min(function(nonzero_indexes)) for function in bands]
            bucket_id = hash(tuple(signature))
            # Add the data index to the current bucket
            self.tables[hash_idx][bucket_id].append(self.cur_data_idx)
            all_bucket_ids.append(bucket_id)
        # Associate the bucket ids with the current data point
        self.data_idx_to_bucket_ids[self.cur_data_idx] = all_bucket_ids
        data_idx = self.cur_data_idx
        # Increment the data index
        self.cur_data_idx += 1
        return data_idx, all_bucket_ids

    def query(self, x, alpha=1, threshold=0):
        nonzero_indexes = np.where(x == 1)[0]
        if len(nonzero_indexes) == 0:
            return
        collision_freqs = defaultdict(int)
        for hash_idx, bands in enumerate(self.hash_functions):
            signature = [np.min(function(nonzero_indexes)) for function in bands]
            bucket_id = hash(tuple(signature))
            for data_index in self.tables[hash_idx][bucket_id]:
                collision_freqs[data_index] += 1
        filtered_collisions = [idx for idx, freq in collision_freqs.items() if freq >= alpha]
        if threshold > 0:
            # filtered_collisions = [idx for idx in filtered_collisions if jaccard()]
            pass
        return filtered_collisions

    def query_idx(self, query_idx, alpha=1, threshold=0):
        '''
        Returns the indices of all hashed data points that collide with the item
        in at least alpha of the t tables. If specified, additionally returns
        only those items with a Jaccard similarity above a threshold
        Args:
            query_idx: the data index of the item to query on
            alpha: the number of collisions in order to return a match
            threshold: the minimum Jaccard similarity in order to return a match
        '''
        bucket_ids = self.data_idx_to_bucket_ids[query_idx]
        collision_freqs = defaultdict(int)
        for hash_idx, bucket_id in enumerate(bucket_ids):
            for data_index in self.tables[hash_idx][bucket_id]:
                if data_index != query_idx:
                    collision_freqs[data_index] += 1
        filtered_collisions = [idx for idx, freq in collision_freqs.items() if freq >= alpha]
        if threshold > 0:
            # filtered_collisions = [idx for idx in filtered_collisions if jaccard()]
            pass
        return filtered_collisions


class AlphaLSHContainer():
    '''A data structure that uses alpha-tunable LSH to compute approximate nearest neighbors.'''
    def __init__(self, t=5, r=1, seed=42):
        self.lsh = AlphaMinhash(t=t, r=r, seed=seed)
        self.data = defaultdict(int)

    def store(self, input_data):
        '''Store input data in the LSH scheme.'''
        input_data_idxs = []
        for row_idx, data_point in tqdm(enumerate(input_data), desc="Hashing data", total=input_data.shape[0]):
            data_idx, bucket_ids = self.lsh.minhash(data_point)
            print(f'Stored data point {data_idx}')
            assert data_idx not in self.data
            self.data[data_idx] = input_data
            input_data_idxs.append(data_idx)
        return input_data_idxs

    def query_idx(self, data_idx, alpha=1, threshold=0):
        return self.lsh.query_idx(data_idx, alpha=alpha, threshold=threshold)

    def query(self, x, alpha=1, threshold=0):
        return self.lsh.query_idx(x, alpha=alpha, threshold=threshold)


if __name__ == "__main__":
    r = 1
    t = 5
    lsh_container = AlphaLSHContainer(t=t, r=1)
    dims = 1000
    n_data = 100
    data = np.random.randint(0, 2, (n_data, dims))
    new_idxs = lsh_container.store(data)
    assert new_idxs == list(range(n_data))
    q_idx = 42
    alphas = range(1, t)
    for alpha in alphas:
        nb_idxs = lsh_container.query_idx(q_idx)
        print(nb_idxs)
        print([1 - jaccard(data[q_idx], data[nb_idx]) for nb_idx in nb_idxs])
    # test_idx = 42
    # bucket_ids = lsh.data_idx_to_bucket_ids[test_idx]
    # collision_idxs = list(set(sum([lsh.tables[hash_idx][bucket_id] for hash_idx, bucket_id in enumerate(bucket_ids)], [])))
    # collision_idxs.remove(test_idx)
    # print(f"Test index {test_idx} collides with {collision_idxs}")
    # print("Average Jaccard similarity of collisions:", np.mean([jaccard(data[test_idx], data[collision_idx]) for collision_idx in collision_idxs]))