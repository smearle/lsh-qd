import math
import os

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.spatial.distance import jaccard  # jaccard distance measure


figs_dir = 'figs'
coll_probs_dir = os.path.join(figs_dir, 'coll_probs')


class MinHash():
    '''
    An implementation of the MinHash algorithm for approximate nearest neighbor search over binary strings.
    Args:
        t: the number of hash tables
        r: the number of bands
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

    def query(self, x, threshold=0):
        '''
        Find the approximate near-neighbors of a given query item x.
        Args:
            x: the query item, a binary string
        Returns:
            nb_idxs (list(int)): a list of indices of items with which the query item collides ("neighbors")
        '''
        collision_freqs = self._get_collision_freqs(x)
        nb_idxs = list(collision_freqs.keys())

        return nb_idxs

    def query_idx(self, query_idx, alpha=1):
        '''
        Like query(), but assumes the item has already been hashed.Returns the indices of all hashed data points that
        collide with the item in at least alpha of the t tables.

        *DEPRECATED* TODO: integrate this incontainer class.
        If specified, additionally returns
        only those items with a Jaccard similarity above a threshold
        Args:
            query_idx: the data index of the item to query on
            alpha: the number of collisions in order to return a match
            threshold: the minimum Jaccard similarity in order to return a match
        Returns:
            nb_idxs (:obj: `list` of :obj: `int`):
        '''
        collision_freqs = self._get_collision_freqs_idx(query_idx)
        nb_idxs = list(collision_freqs.keys())

        return nb_idxs

    def _get_collision_freqs(self, x):
        '''
        Args:
            query_idx:
        Returns:
            collision_freqs (dict{int: int})): a dictionary mapping the index of each neighbor to the number of buckets
                in which it collided with the query item (always less than or equal to t, the # of total tables)
        '''
        nonzero_indexes = np.where(x == 1)[0]

        if len(nonzero_indexes) == 0:
            return

        collision_freqs = defaultdict(int)

        for hash_idx, bands in enumerate(self.hash_functions):

            # "AND" operation over r bands to get the low-dimensional projection of each item
            signature = [np.min(function(nonzero_indexes)) for function in bands]

            # Hash the resulting signature into each of t tables
            bucket_id = hash(tuple(signature))

            # Keep a count of number-of-collisions with each item
            for data_index in self.tables[hash_idx][bucket_id]:
                collision_freqs[data_index] += 1

        return collision_freqs


    def _get_collision_freqs_idx(self, query_idx):
        '''
        Gets the collision frequencies of an item with items already hashed to the scheme. Assumes the query item has
        already been hashed.
        Returns:
            collision_freqs (dict(int: int)): a dictionary mapping the index of each neighbor to the number of buckets
                in which it collided with the query item (always less than or equal to t, the # of total tables)
        '''
        bucket_ids = self.data_idx_to_bucket_ids[query_idx]
        collision_freqs = defaultdict(int)

        for hash_idx, bucket_id in enumerate(bucket_ids):
            for data_index in self.tables[hash_idx][bucket_id]:
                if data_index != query_idx:
                    collision_freqs[data_index] += 1

        return collision_freqs


class AlphaMinHash(MinHash):
    '''
    An implementation of the MinHash algorithm for approximate nearest
    neighbors on binary strings, designed for use in niching and QD. In
    addition to the typical parameters of # of tables and # of bands, this
    also accepts an alpha parameter when querying. An item in the database
    is returned only if it is hashed into the same bucket as the query in
    at least alpha of the t tables.
    Args:
        t: the number of distinct hash tableks
        r: the number of bands (i.e. hash functions) used to determine an item's
            location in each of the t tables
        seed: the random seed for generating the hash functions
    '''

    def __init__(self,
                 t,
                 r,
                 seed=42):
        super(AlphaMinHash, self).__init__(t=t, r=r, seed=seed)

    def query(self, x, alpha=1):
        '''
        Args:
            alpha (int): minimum number of tables in which an item must collide with x in order t be considered a
                neighbor
        Returns:
            nb_idxs (list(int)): a list of indices of neighbor items
        '''
        collision_freqs = super()._get_collision_freqs(x)
        nb_idxs = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return nb_idxs

    def query_idx(self, query_idx, alpha=1):
        '''
        Returns the indices of all hashed data points that collide with the item
        in at least alpha of the t tables. If specified, additionally returns
        only those items with a Jaccard similarity above a threshold
        Args:
            query_idx: the data index of the item to query on
            alpha: the number of collisions in order to return a match
            threshold: the minimum Jaccard similarity in order to return a match
        '''
        collision_freqs = super()._get_collision_freqs_idx(query_idx)
        filtered_collisions = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return filtered_collisions


class AlphaLSHContainer():
    '''A data structure that uses alpha-tunable LSH to compute approximate nearest neighbors.'''
    def __init__(self, t=5, r=1, seed=42):
        self.lsh = AlphaMinHash(t=t, r=r, seed=seed)
        self.data = defaultdict(int)

    def store(self, input_data):
        '''
        Store a batch of input data in the LSH scheme. The LSH scheme only stores data indices, which this container
        stores the actual datapoints.
        Args:
            input_data: a (n_data, n_dims)-shape array of new datapoints to be stored in the container
        '''
        input_data_idxs = []
        for row_idx, data_point in tqdm(enumerate(input_data), desc="Hashing data", total=input_data.shape[0]):
            data_idx, bucket_ids = self.lsh.minhash(data_point)
            assert data_idx not in self.data
            self.data[data_idx] = data_point
            input_data_idxs.append(data_idx)

        return input_data_idxs

    def query_idx(self, data_idx, alpha=1, threshold=0):
        return self.lsh.query_idx(data_idx, alpha=alpha)

    def query(self, x, alpha=1, threshold=0):
        return self.lsh.query_idx(x, alpha=alpha)


def collision_prob(sim, r, t):
    '''
    Compute the collisiion probabilities between pairs of items with given similarity scores (or inverse distances)
    between them.
    Args:
        sim: an integer or 1D vector of similarities between in [0, 1]
        r (int): the number of bands in each LSH instance in the hashing scheme
        t (int): the number of tables (LSH instances) in the hashing scheme
    Returns:
        an integer or 1D vector of collision probabilities in [0, 1]
    '''
    p_coll = 1 - (1 - sim ** r) ** t

    return p_coll

def collision_prob_alpha(sim, r, t, alpha=1):
    '''
    Compute the collision probabities given similarity scores for a given value of alpha in the alpha-tunable
    LSH scheme.
    Args:
        sim: an integer or 1D vector of similarities between in [0, 1]
        r (int): the number of bands in each LSH instance in the hashing scheme
        t (int): the number of tables (LSH instances) in the hashing scheme
    Returns:
        an integer or 1D vector of collision probabilities in [0, 1]
    '''
    p_coll = 0

    for i in range(int(alpha), t + 1):
        p_coll += math.comb(t, i) * (sim ** r) ** i * (1 - sim ** r) ** (t - i)

    if alpha == 1:

        # Ensure we achieve the same s-curve as the equivalent vanilla LSH scheme when alpha=1
        vanilla_p_coll = collision_prob(sim, r, t)

        # There can be small discrepancies here
        assert np.abs(p_coll - vanilla_p_coll).max() < 1e-5

    return p_coll

def plot_collision_prob(rs, ts):
    '''
    Plot collision probabilities by similarity in an LSH scheme over a set of parameters.
    Args:
        rs: 1D array of r values (number of bands)
        ts: 1D array of t values (number of tables)
    '''
    cps = []
    sim = np.arange(0, 1.01, 1 / 100)  # the
    for t in ts:
        r_cps = []
        for ri, r in enumerate(rs):
            cp = collision_prob(sim, r, t)
            r_cps.append(cp)
            label = f'r = {int(r)}' if ri == 0 or ri == len(rs) - 1 else None
            plt.plot(sim, cp, label=label, color='red', alpha=0.2 + 0.8 * (ri / (len(rs) - 1)))
        cps.append(r_cps)
        plt.legend()
        plt.xlabel('item similarity')
        plt.ylabel('collision probability')
        plt.savefig(os.path.join(f'{coll_probs_dir}', f't-{t}'))
        plt.close()

def plot_collision_prob_alpha(r, t):
    '''
    Plots the collision probability by distance of a tunable LSH scheme with given parameters.
    Args:
        r: number of bands in each LSH instance
        t: number of LSH instances
    '''
    v = np.arange(0, 1.01, 1 / 100)  # the
    alphas = np.arange(0, t + 1, t / 20)
    for a in alphas:
        p_coll = collision_prob_alpha(v, r, t, alpha=a)
        label = f'alpha = {int(a)}' if a == alphas[0] or a == alphas[-1] else None
        plt.plot(v, p_coll, label=label, color='blue', alpha=0.2 + 0.8 * (a / (alphas[-1] - alphas[0])))
    plt.xlabel('item similarity')
    plt.ylabel('collision probability')
    plt.legend()
    plt.savefig(f'{coll_probs_dir}/alphas_r-{r}_t-{t}.png')
    plt.close()

def plot_mean_nbs_by_alpha():
    '''
    Plot the effect of the LSH alpha parameter on the average neighborhood size and intra-neighborhood similarity in
     neighborhoods returned by the LSH scheme.
    '''
    r = 1
    t = 12
    lsh_container = AlphaLSHContainer(t=t, r=r)
    dims = 1000
    n_data = 100
    data = np.random.randint(0, 2, (n_data, dims))
    new_idxs = lsh_container.store(data)
    assert new_idxs == list(range(n_data))
    n_queries = 10
    q_idxs = np.random.choice(list(lsh_container.data.keys()), n_queries, replace=False)
    alphas = range(1, t)
    # TODO: Is this equivalent to just averaging over *all* alpha-neighborhoods in our container? Is there an easy way
    #   to collect all these neighborhoods?
    a_n_nbs = []  # mean neighborhood size per alpha
    a_sims = []  # mean similarity within a neighborhood per alpha
    for alpha in alphas:
        q_n_nbs = []  # neighborhood size for each query item
        q_sims = []  # mean intra-neighborhood similarity for each query item
        for q_idx in q_idxs:
            nb_idxs = lsh_container.query_idx(q_idx, alpha=alpha)
            n_nbs = len(nb_idxs)
            mean_sim = np.mean([1 - jaccard(lsh_container.data[i], lsh_container.data[j]) for i in nb_idxs
                               for j in nb_idxs if i > j])
            q_n_nbs.append(n_nbs)
            q_sims.append(mean_sim)
            # print(nb_idxs)
            # print([1 - jaccard(lsh_container.data[q_idx], lsh_container.data[nb_idx]) for nb_idx in nb_idxs])
        a_n_nbs.append(np.mean(q_n_nbs))
        a_sims.append(np.mean(q_sims))

    plt.plot(alphas, a_n_nbs)
    plt.ylabel('mean number of neighbors')
    plt.xlabel('alpha')
    plt.savefig(f'{figs_dir}/mean_nbs_X_alpha')
    plt.close()

    # a_sims = np.where(np.isnan(a_sims), 1, a_sims)
    plt.plot(alphas, a_sims)
    plt.ylabel('mean intra-neighborhood similarity')
    plt.xlabel('alpha')
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/mean_sim_X_alpha')
    plt.close()

    # test_idx = 42
    # bucket_ids = lsh.data_idx_to_bucket_ids[test_idx]
    # collision_idxs = list(set(sum([lsh.tables[hash_idx][bucket_id] for hash_idx, bucket_id in enumerate(bucket_ids)], [])))
    # collision_idxs.remove(test_idx)
    # print(f"Test index {test_idx} collides with {collision_idxs}")
    # print("Average Jaccard similarity of collisions:", np.mean([jaccard(data[test_idx], data[collision_idx]) for collision_idx in collision_idxs]))


if __name__ == "__main__":
    plot_mean_nbs_by_alpha()
    rs = np.arange(1, 20)
    ts = np.arange(20, 21)
    plot_collision_prob(rs=rs, ts=ts)
    plot_collision_prob_alpha(r=1, t=20)