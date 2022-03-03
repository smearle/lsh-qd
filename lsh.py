import math
import numpy as np

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
            proj_values = np.array([fn(y) for fn in band_funcs])
            signature = np.floor(proj / self.r)
            bucket_id = tuple(signature)

            # Add the data index to the current bucket
            self.tables[hash_idx][bucket_id].append(self.cur_data_idx)
            all_bucket_ids.append(bucket_id)

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

            proj_values = np.array([fn(y) for fn in band_funcs])
            signature = np.floor(proj / self.r)
            bucket_id = tuple(signature)

            # Keep a count of number of collisions with each other item
            for data_index in self.tables[hash_idx][bucket_id]:
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

        collision_freqs = self.lsh._get_collision_freqs(y)
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

        collision_freqs = self.lsh._get_collision_freqs(y)
        neighbor_idxs = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return neighbor_idxs


class MultiProbeLsh(pStableHash):
    '''
    TODO
    '''
    def __init__(self, k, l, r, n_dims, seed):
        super().__init__(k, l, r, n_dims, seed)

    def query(self, y, num_perturbations):

        projection_values = []
        bucket_ids = []

        # Compute the signature of y under each of our l tables
        for hash_idx, band_funcs in enumerate(self.hash_functions):

            proj_values = np.array([fn(y) for fn in band_funcs])
            signature = np.floor(proj / self.r)
            bucket_id = tuple(signature)

            bucket_ids.append(bucket_id)


