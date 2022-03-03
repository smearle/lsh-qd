import math
import heapq
import numpy as np
from collections import defaultdict

class pStableHash():
    """
        An instance of a p-stable hashing scheme, which can be used for classic LSH, MultiProbeLSH, AlphaLSH, etc.

        Args:
            k: number of bands per hash function
            l: number of hash functions/tables
            r: the width of the intervals for each projection
            n_dims: the number of dimensions in our dataset, required to project data points to random lines
            seed: the random seed
        """
    def __init__(self, k, l, r, n_dims, seed=42):
        
        self.k, self.l, self.r = k, l, r
        self.n_dims = n_dims
        self.seed = seed

        # We use the largest 32-bit integer as our maximum hash value, but this could be changed without too much effort
        self.max_val = 2 ** 32 - 1
        self.p = 4294967311

        # Store each of hash function in an array of size (l, k). self.hash_functions[i][j] indicates the j'th band of the 
        # i'th table
        self.hash_functions = [[self._generate_hash_function() for _ in range(self.k)]
                               for _ in range(self.l)]

        # This dictionary maps from a data index to a list of l bucket ids (the hash of vector of k integers) indicating the 
        # keys in each hash table where the index can be found
        self.cur_data_idx = 0
        self.data_idx_to_bucket_ids = {}

        # This list stores l dictionaries, each of which maps from a hash value / bucket id to each of the data indexes which 
        # have been hashed to that bucket
        self.tables = [defaultdict(list) for _ in range(self.l)]

    def _generate_hash_function(self):
        '''
        This function should return another function which takes as input a data point (vector) and returns a hash value (an 
        integer). A vector of k such integers defines a data point's "signature" in that table, and the hash value of that 
        signature is the bucket ID
        '''

        # Generate the random projection vector
        a = np.random.normal(0, 1, size=self.n_dims)

        # Sample some random offset to nullify in expectation the effect of bin border placement
        b = np.random.random() * self.r


        # The hash function here actually only projects the item onto a random line, it doesn't assign it to a 
        # random interval. That responsibiltiy belongs to hash() and _get_collision_freqs(), which need to divide
        # the projected value by self.r and apply a floor operator
        def hash_func(x):
            return x.T @ a + b

        return hash_func

    def hash(self, x):
        '''
        Hashes a data point, obtaining a hash value for each of the l tables (based on its value under each of the k bands).
        Returns the 'data index' of the data point as well as the l bucket IDs

        Args:
            x: the data point
        '''
        all_bucket_ids = []

        # For each table, collect the k hash functions (one for each band)
        for hash_idx, band_funcs in enumerate(self.hash_functions):

            # Compute the data point's signature under those hash functions, and its corresponding bucket id
            table_proj_values = np.array([fn(x) for fn in band_funcs])
            table_bucket_id = tuple(np.floor(table_proj_values / self.r).astype(np.int32))

            # Add the data index to the current bucket
            self.tables[hash_idx][table_bucket_id].append(self.cur_data_idx)
            all_bucket_ids.append(table_bucket_id)

        # Associate the bucket ids with the current data point
        self.data_idx_to_bucket_ids[self.cur_data_idx] = all_bucket_ids
        data_idx = self.cur_data_idx

        # Increment the data index
        self.cur_data_idx += 1

        return data_idx, all_bucket_ids

    def query(self, y):
        '''
        Return all of the previously-hashed approximate near neighbors to a provided query point
        '''
        raise NotImplementedError

    def get_collision_prob(self, similarity):
        '''
        Return the expected probability that the hashing scheme will return an item with specified 
        similarity to an arbitrary query point

        TODO: similarity can techincally represent different quanitites in different domains / contexts. We
            should think about how we wan to define the arguments for this function (i.e. should it take raw 
            distances? actualy pairs of points? etc.)
        '''
        raise NotImplementedError


    def _get_collision_freqs(self, y):
        '''

        For a query data point y, return a dictionary that maps from data indexes to the number of times that
        data point collides with the query point (an integer between 0 and l). 

        NOTE: if y is a data point that has been previously hashed, then this function will include its data
            index, with l collisions

        Args:
            y: a query point
        '''
        collision_freqs = defaultdict(int)

        for hash_idx, band_funcs in enumerate(self.hash_functions):

            table_proj_values = np.array([fn(y) for fn in band_funcs])
            table_bucket_id = tuple(np.floor(table_proj_values / self.r).astype(np.int32))

            # Keep a count of number of collisions with each other item
            for data_index in self.tables[hash_idx][table_bucket_id]:
                collision_freqs[data_index] += 1

        return collision_freqs

class VanillaLSH(pStableHash):
    '''
    TODO

    Args:
            k: number of bands per hash function
            l: number of hash functions/tables
            r: the width of the intervals for each projection
            n_dims: the number of dimensions in our dataset, required to project data points to random lines
            seed: the random seed
    '''

    def __init__(self, k, l, r, n_dims, seed):
        super().__init__(k, l, r, n_dims, seed)

    def query(self, y):
        '''
        Returns the data indices of each previously-hashed item which collides with query point in at least one
        of the l tables

        Args:
            y: a query_point
        '''

        collision_freqs = self._get_collision_freqs(y)
        neighbor_idxs = list(collision_freqs.keys()) # the collision_freqs dict only includes elements that collide at least once

        return neighbor_idxs


class AlphaLSH(pStableHash):
    '''
    An extension of LSH algorithms for approximate nearest neighbors, designed for use in niching and QD. In addition to 
    the typical parameters of # of tables and # of bands, this also accepts an alpha parameter when querying. An item in 
    the database is returned only if it is hashed into the same bucket as the query in at least alpha of the l tables.

    Args:
            k: number of bands per hash function
            l: number of hash functions/tables
            r: the width of the intervals for each projection
            n_dims: the number of dimensions in our dataset, required to project data points to random lines
            seed: the random seed
    '''

    def __init__(self, k, l, r, n_dims, seed):
        super().__init__(k, l, r, n_dims, seed)

    def query(self, y, alpha=1):
        '''
        Returns the data indices of each previously-hashed item which collides with the query point in at least
        alpha of the l tables

        Args:
            y: a query point
            alpha (int): minimum number of tables in which an item must collide with x in order l be considered a
                neighbor
        '''

        collision_freqs = self._get_collision_freqs(y)
        neighbor_idxs = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return neighbor_idxs


class MultiProbeLsh(pStableHash):
    '''
    TODO
    '''
    def __init__(self, k, l, r, n_dims, seed):
        super().__init__(k, l, r, n_dims, seed)

    def query(self, y, num_perturbations):

        bucket_ids = []
        negative_boundary_dists = []
        positive_boundary_dists = []

        # Compute the bucket id of y under each of our l tables
        for hash_idx, band_funcs in enumerate(self.hash_functions):

            table_proj_values = np.array([fn(y) for fn in band_funcs])
            table_bucket_id = np.floor(table_proj_values / self.r).int()

            # Compute the distance from the query point to the positive and negative boundaries
            # of its interval, which will be used to determine the ordering of perturbation vectors
            table_negative_boundary_dists = table_proj_values - (table_bucket_id * self.r)
            table_positive_boundary_dists = self.r - table_negative_boundary_dists

            bucket_ids.append(tuple(table_bucket_id))
            negative_boundary_dists.append(table_negative_boundary_dists)
            positive_boundary_dists.append(table_positive_boundary_dists)


        # Negative / positive boundary dists are l by k. Ultimately, we want a matrix X to be (l, k, 3)
        zeros = np.zeros(self.l, self.k)
        negative_boundary_dists = np.array(negative_boundary_dists)
        positive_boundary_dists = np.array(positive_boundary_dists)

        # This matrix stores all of the distances from the query point to its interval boundaries. 
        # X[i][j][delta] represents the distance in the i'th table and the j'th band from the query
        # point to the boundary of the bucket that is delta away (though X[i][j][0] always = 0)
        X = np.stack([zeros, positive_boundary_dists, negative_boundary_dists], axis=2)

        # This matrix stores all of the distances from the query point to tis interval boundaries.
        # X[i, j] represents the distance from in the i
        X = np.concatenate([negative_boundary_dists, positive_boundary_dists], axis=1)

        sorted_indices = np.argsort(X, axis=1)

        # The above indices are from 0 to 2k-1, and here we create two new arrays which give us
        # the band index and the actual perturbation, in ascending order of score along axis 1
        sorted_band_idxs = sorted_indices % self.k
        sorted_perturbations = (sorted_indices // self.k * 2) - 1


        # Need opertaion that maps from index in [0, 2k-1] to {-1, 1} X {0, ..., k-1}:
        # f(idx): (idx // k * 2) - 1, idx % k


        # Create a min-heap to store all of our potential perturbations. Each entry in the head will look
        # like (score, table_idx, [(band_idx, perturbation), (band_idx, perturbation), ...]). To begin,
        # the heap will contain (score, table_idx, [(best_perturbation_idx, best_perturbation)]) for each
        # table_idx in {0, ..., l-1}
        heap = [(X[table_idx][sorted_indices[table_idx][0]], table_idx, [(sorted_band_idxs[table_idx][sorted_indices[table_idx][0]], sorted_perturbations[table_idx][sorted_indices[table_idx][0]])]) for table_idx in range(self.l)]

        heap = []
        for table_idx in range(self.l):
            best_band_and_perturbation_idx = sorted_indices[table_idx][0] # a value between 0 and 2k-1
            best_band_idx = sorted_band_idxs[table_idx][0] # a value between 0 and k-1
            best_perturbation = sorted_perturbations[table_idx][0] # -1 or 1

            score = X[table_idx][best_band_and_perturbation_idx]

            heap.append((score, table_idx, [(best_band_idx, best_perturbation)]))

        heapq.heapify(heap)

        # X is [l, 2k] where the first k elements are delta = 1 for each band, and the last k are delta = -1
        # i \in [0, 2k-1] --> [1, j] or [-1, j']

        sorted_index = np.argsort(X, axis=2)

if __name__ == "__main__":
    k = 5
    l = 10
    r = 1
    n_dims = 100
    seed = 42

    vanilla = VanillaLSH(k, l ,r, n_dims, seed)
    alpha = AlphaLSH(k, l ,r, n_dims, seed)
    multi = MultiProbeLsh(k, l ,r, n_dims, seed)

    point = np.random.random(n_dims)

    print(vanilla.hash(point))
    print(alpha.hash(point))
    print(multi.hash(point))

    print("\n\n")
    print(vanilla.query(point))
