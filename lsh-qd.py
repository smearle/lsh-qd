import math
import os

import numpy as np
from scipy import integrate
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.spatial.distance import jaccard  # jaccard distance measure


figs_dir = 'figs'
coll_probs_dir = os.path.join(figs_dir, 'coll_probs')


class LocalitySensitiveHash():
    def __init__(self, k, l, seed=42):
        '''
        A locality sensitive hashing scheme in the abstract.
        Args:
            k: number of bands per hash function
            l: number of hash functions/tables
        '''


        self.k, self.l = k, l
        self.seed = seed

        # We use the largest 32-bit integer as our maximum hash value, but
        # this could be changed without too much effort
        self.max_val = 2 ** 32 - 1
        self.p = 4294967311

        # Store each of hash function in an array of size (l, k). self.hash_functions[i][j]
        # indicates the j'th band of the i'th table
        self.hash_functions = [[self._generate_hash_function() for _ in range(self.k)]
                               for _ in range(self.l)]

        # This dictionary maps from a data index to a list of l bucket ids
        # indicating the keys in each hash table where the index can be found
        self.cur_data_idx = 0
        self.data_idx_to_bucket_ids = {}

        # This list stores l dictionaries, each of which maps from a hash value /
        # bucket id to each of the data indexes which have been hashed to that bucket
        self.tables = [defaultdict(list) for _ in range(self.l)]

    def hash(self, x):
        '''
        Args:
            x:
        Returns:
        '''
        all_bucket_ids = []

        # For each set of k bands...
        for hash_idx, band_funcs in enumerate(self.hash_functions):
            # Compute the data point's signature under those hash functions, and its
            # corresponding bucket id
            signature = self._get_signature(x, band_funcs)
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
        collide with the item in at least alpha of the l tables.

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
            x: (:obj: `np.array' of :obj: int)
        Returns:
            collision_freqs (dict{int: int})): a dictionary mapping the index of each neighbor to the number of buckets
                in which it collided with the query item (always less than or equal to l, the # of total tables)
        '''
        collision_freqs = defaultdict(int)

        for hash_idx, band_funcs in enumerate(self.hash_functions):

            # "AND" operation over k bands to get the low-dimensional projection of each item
            signature = self._get_signature(x, band_funcs=band_funcs)

            # Hash the resulting signature into each of l tables
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
                in which it collided with the query item (always less than or equal to l, the # of total tables)
        '''
        bucket_ids = self.data_idx_to_bucket_ids[query_idx]
        collision_freqs = defaultdict(int)

        for hash_idx, bucket_id in enumerate(bucket_ids):
            for data_index in self.tables[hash_idx][bucket_id]:
                if data_index != query_idx:
                    collision_freqs[data_index] += 1

        return collision_freqs


class MinHash(LocalitySensitiveHash):
    '''
    An implementation of the MinHash algorithm for approximate nearest neighbor search over binary strings.
    Args:
        k: the number of bands
        l: the number of hash tables
    '''

    def __init__(self,
                 k,
                 l,
                 seed=42):

        super().__init__(k, l, seed=seed)

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

    def _get_signature(self, nonzero_indexes, band_funcs):
        signature = [np.min(function(nonzero_indexes)) for function in band_funcs]

        return signature

    def hash(self, x):
        '''
        Performs the Minhash algorithm on the provided binary string, hashing the data
        into each of the l tables. Also returns the hash indices
        Args:
            x: the input data, a binary string
        '''
        nonzero_indexes = np.where(x == 1)[0]
        if len(nonzero_indexes) == 0:
            return

        data_idx, all_bucket_ids = super().hash(nonzero_indexes)

        return data_idx, all_bucket_ids


    def _get_collision_freqs(self, x):
        '''
        Args:
            x (:obj: `np.array' of :obj: int): a binary vector
        Returns:
            collision_freqs (dict{int: int})): a dictionary mapping the index of each neighbor to the number of buckets
                in which it collided with the query item (always less than or equal to l, the # of total tables)
        '''
        nonzero_indexes = np.where(x == 1)[0]
        collision_freqs = super()._get_collision_freqs(nonzero_indexes)
        if len(nonzero_indexes) == 0:
            return

        return collision_freqs


class pStableHash(LocalitySensitiveHash):
    def __init__(self,
                 k,
                 l,
                 r=4,
                 n_dims=1000,
                 seed=42):
        '''
        p-stable hashing for approximate nearest neighbor search over real-values vectors.
        Args:
            k: number of projections per hash function
            l: number of hash functions/tables
            r: the width of buckets for each projection (each "band")
        '''
        self.r = r
        self.n_dims = n_dims
        super().__init__(k, l, seed=seed)

    def _generate_hash_function(self):

        # Generate the random projection vector
        a = np.random.normal(0, 1, size=self.n_dims)

        # Sample some random offset to nullify in expectation the effect of bin border placement
        b = np.random.random() * self.r

        def hash_func(x):
            # Project the item to a line and discretize the line into buckets; return bucket id
            return math.floor((x.T @ a + b) / self.r)

        return hash_func

    def _get_signature(self, x, band_funcs):
        '''
        Args:
            x: item representation as real-valued vector
            band_funcs: hash functions of the type generated in self._generate_hash_function
        '''
        signature = [fn(x) for fn in band_funcs]

        return signature


class AlphaLSH(object):
    '''
    An extension of LSH algorithms for approximate nearest
    neighbors, designed for use in niching and QD. In
    addition to the typical parameters of # of tables and # of bands, this
    also accepts an alpha parameter when querying. An item in the database
    is returned only if it is hashed into the same bucket as the query in
    at least alpha of the l tables.
    Args:
        t: the number of distinct hash tables
        r: the number of bands (i.e. hash functions) used to determine an item's
            location in each of the l tables
        seed: the random seed for generating the hash functions
    '''

    def __init__(self,
                 k,
                 l,
                 lsh_cls=MinHash,
                 seed=42,
                 **lsh_args):
        self.lsh = lsh_cls(k=k, l=l, seed=seed, **lsh_args)

    def query(self, x, alpha=1):
        '''
        Args:
            alpha (int): minimum number of tables in which an item must collide with x in order l be considered a
                neighbor
        Returns:
            nb_idxs (list(int)): a list of indices of neighbor items
        '''
        collision_freqs = self.lsh._get_collision_freqs(x)
        nb_idxs = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return nb_idxs

    def query_idx(self, query_idx, alpha=1):
        '''
        Returns the indices of all hashed data points that collide with the item
        in at least alpha of the l tables. If specified, additionally returns
        only those items with a Jaccard similarity above a threshold
        Args:
            query_idx: the data index of the item to query on
            alpha: the number of collisions in order to return a match
            threshold: the minimum Jaccard similarity in order to return a match
        '''
        collision_freqs = self.lsh._get_collision_freqs_idx(query_idx)
        filtered_collisions = [idx for idx, freq in collision_freqs.items() if freq >= alpha]

        return filtered_collisions

    def hash(self, x):
        return self.lsh.hash(x)


class AlphaLSHContainer():
    '''A data structure that uses alpha-tunable LSH to compute approximate nearest neighbors.'''
    def __init__(self,
                 k=1,
                 l=5,
                 lsh_cls=AlphaLSH,
                 lsh_args={},
                 seed=42):
        self.lsh = lsh_cls(k=k, l=l, seed=seed, **lsh_args)
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
            data_idx, bucket_ids = self.lsh.hash(data_point)
            assert data_idx not in self.data
            self.data[data_idx] = data_point
            input_data_idxs.append(data_idx)

        return input_data_idxs

    def query_idx(self, data_idx, alpha=1, threshold=0):
        return self.lsh.query_idx(data_idx, alpha=alpha)

    def query(self, x, alpha=1, threshold=0):
        return self.lsh.query(x, alpha=alpha)


def collision_prob(sim, k, l):
    '''
    Compute the collisiion probabilities between pairs of items with given similarity scores (or inverse distances)
    between them.
    Args:
        sim: an integer or 1D vector of similarities between in [0, 1]
        k (int): the number of bands in each LSH instance in the hashing scheme
        l (int): the number of tables (LSH instances) in the hashing scheme
    Returns:
        an integer or 1D vector of collision probabilities in [0, 1]
    '''
    p_coll = 1 - (1 - sim ** k) ** l

    return p_coll

def collision_prob_alpha(sim, k, l, alpha=1):
    '''
    Compute the collision probabities given similarity scores for a given value of alpha in the alpha-tunable
    LSH scheme.
    Args:
        sim: an integer or 1D vector of similarities between in [0, 1]
        k (int): the number of bands in each LSH instance in the hashing scheme
        l (int): the number of tables (LSH instances) in the hashing scheme
    Returns:
        an integer or 1D vector of collision probabilities in [0, 1]
    '''
    p_coll = 0

    for i in range(int(alpha), l + 1):
        p_coll += math.comb(l, i) * (sim ** k) ** i * (1 - sim ** k) ** (l - i)

    if alpha == 1:

        # Ensure we achieve the same s-curve as the equivalent vanilla LSH scheme when alpha=1
        vanilla_p_coll = collision_prob(sim, k, l)

        # There can be small discrepancies here
        assert np.abs(p_coll - vanilla_p_coll).max() < 1e-5

    return p_coll

def collision_prob_pstable(dists, k, l, r, f_G):
    '''
    Args:
        sim:
        k: number of projections (bands)
        l: number of hash tables
        r: the width of the buckets in the projections
    Returns:
        an integer or 1D vector of collision probabilities in [0, 1]
    '''
    x_res = 1000
    p_projs = np.empty(len(dists))

    assert dists[0] == 0
    p_projs[0] = 1
    for di, d in enumerate(dists[1:]):
        result = integrate.quad(lambda t: (1 / d) * f_G(t / d) * (1 - (t / r)), 0, r)
        p_proj = 2 * result[0]
        p_projs[di+1] = p_proj

    coll_probs = collision_prob(sim=p_projs, k=k, l=l)

    return coll_probs


def plot_collision_prob_pstable(ks, ls, rs):

    # d = np.linspace(-2, 2, 100)
    #
    def f_G(x):
        return np.exp(-x**2/2) / math.sqrt(2*math.pi)
    # f_G_d = f_G(d)
    # plt.plot(d, f_G_d)
    # plt.show()

    dists = np.linspace(0, 4, 100)
    for li, l in enumerate(ls):
        for ri, r in enumerate(rs):
            for ki, k in enumerate(ks):
                coll_probs = collision_prob_pstable(dists, k, l, r, f_G)
                label = f'k = {round(k, 2)}' if ki == 0 or ki == len(ks) - 1 else None
                plt.plot(dists, coll_probs, label=label, color='blue', alpha=1 - 0.8 * (len(ks) - ki) / len(ks))
            plt.xlabel('item distances')
            plt.ylabel('collision probability')
            plt.legend()
            plt.savefig(os.path.join(f'{coll_probs_dir}', f'pstable_l-{l}_r-{r}'))
            plt.close()

def plot_collision_prob(ks, ls):
    '''
    Plot collision probabilities by similarity in an LSH scheme over a set of parameters.
    Args:
        ks: 1D array of k values (number of bands)
        ls: 1D array of l values (number of tables)
    '''
    cps = []
    sim = np.arange(0, 1.01, 1 / 100)  # the

    for l in ls:
        l_cps = []

        for ki, k in enumerate(ks):
            cp = collision_prob(sim, k, l)
            l_cps.append(cp)
            label = f'k = {int(k)}' if ki == 0 or ki == len(ks) - 1 else None
            plt.plot(sim, cp, label=label, color='red', alpha=0.2 + 0.8 * (ki / (len(ks) - 1)))

        cps.append(l_cps)
        plt.legend()
        plt.xlabel('item similarity')
        plt.ylabel('collision probability')
        plt.savefig(os.path.join(f'{coll_probs_dir}', f'l-{l}'))
        plt.close()

def plot_collision_prob_alpha(k, l):
    '''
    Plots the collision probability by distance of a tunable LSH scheme with given parameters.
    Args:
        r: number of bands in each LSH instance
        t: number of LSH instances
    '''
    v = np.arange(0, 1.01, 1 / 100)  # the
    alphas = np.arange(0, l + 1, l / 20)

    for a in alphas:
        p_coll = collision_prob_alpha(v, k, l, alpha=a)
        label = f'alpha = {int(a)}' if a == alphas[0] or a == alphas[-1] else None
        plt.plot(v, p_coll, label=label, color='blue', alpha=0.2 + 0.8 * (a / (alphas[-1] - alphas[0])))

    plt.xlabel('item similarity')
    plt.ylabel('collision probability')
    plt.legend()
    plt.savefig(f'{coll_probs_dir}/alphas_k-{k}_l-{l}.png')
    plt.close()

def gen_uni_rand_data_bin(n_data, n_dims):
    '''
    Uniform randomly generate some binary-valued synthetic data for testing the LSH algorithms.
    Args:
        n_data: number of data points
        n_dims: number of features
    Returns:
        data: a 2D (n_data, dims) numpy array of data
    '''
    data = np.random.randint(0, 2, (n_data, n_dims))

    return data

def gen_uni_rand_data_real(n_data, n_dims):
    '''
    Uniform randomly generate some real-valued synthetic data for testing the LSH algorithms, within the unit hypercube.
    Args:
        n_data: number of data points
        n_dims: number of features
    Returns:
        data: a 2D (n_data, dims) numpy array of data
    '''
    data = np.random.random((n_data, n_dims))
    data = (data - 0.5) * 2

    return data

def gen_planted_rand_data_real(query_data, n_data, R, epsilon):
    '''
    Args:
        query_data: the 2D array of query points around which to adversarially construct our data
        R: the radius of interest around each query point
        epsilon: the annulus between R and R + epsilon is the annulus of disinterest
    Returns:
        data: a 2D (n_data, dims) numpy array of data
    '''
    n_dims = query_data.shape[1]
    n_queries = query_data.shape[0]
    n_nbs = n_queries
    n_non_nbs = n_data - n_nbs

    # generate random direction vectors to be projected to the surface of some sphere
    data = np.random.normal(0, 1, (n_data, n_dims))

    # randomly project each neighbor vector to the surface of a ball with uniformly random radius in [0, R]
    # project non-neighbors to the annulus in [R, R+epsilon]
    data = data / np.linalg.norm(data, axis=-1, keepdims=True)
    # random radius sizes for each datapoint
    radii = np.random.random(size=(n_data, 1))
    radii[:n_nbs] *= R  # neighbor radii (of ball)
    radii[n_nbs:] *= epsilon # non-neighbor radii (of annulus)
    data[:n_nbs] *= radii[:n_nbs]  # neighbors are scaled by their radii to be inside ball
    data[n_nbs:] = data[n_nbs:] * R + data[n_nbs] * radii[n_nbs:]  # non-neighbors are scaled by width of annulus then added to themselves
                                                 # at the unit ball

    # re-center the neihbors around each of the query points
    data[:n_nbs] = data[:n_nbs] + query_data

    # translate the non-neighbors to center around random query points
    q_idxs = np.random.choice(n_queries, size=n_non_nbs)
    # FIXME: some non-neighbors fall within the R-balls! Why?
    data[n_nbs:] = data[n_nbs:] + query_data[q_idxs]

    dists = np.linalg.norm(query_data[None, :, :] - data[:, None, :], axis=-1)

    return data

def plot_pairwise_dist(data):
    dists = np.linalg.norm(data[None, :, :] - data[:, None, :], axis=-1)
    n_buckets = 10
    low, high = data.mean() - 1.0 * data.std(), data.mean() + 1.0 * data.std()
    buck_size = (high - low) / n_buckets
    ys = []
    for i in range(n_buckets):
        ys.append(np.sum(np.where((dists >= low + buck_size * i) & (dists < low + buck_size * (i + 1)))))

    plt.plot(ys)
    plt.savefig(os.path.join(f'{figs_dir}', 'planted_data_dist'))
    plt.close()


def plot_mean_nbs_by_alpha(scheme='MinHash'):
    '''
    Plot the effect of the LSH alpha parameter on the average neighborhood size and intra-neighborhood similarity in
    neighborhoods returned by the LSH scheme.
    Args:
        scheme: which type of LSH scheme to use
    '''
    k = 1
    l = 12
    n_data = 100
    n_dims = 100
    n_queries = 10

    if scheme == 'MinHash':
        lsh_container = AlphaLSHContainer(l=l, k=k)
        data = gen_uni_rand_data_bin(n_data, n_dims)
        similarity = lambda a, b: 1 - jaccard(a, b)
        query_data = gen_uni_rand_data_bin(n_queries, n_dims)

    elif scheme == 'pStable':
        lsh_args = {
            'lsh_cls': pStableHash,
            'n_dims': n_dims}
        lsh_container = AlphaLSHContainer(l=l, k=k, lsh_cls=AlphaLSH, lsh_args=lsh_args)
        data = gen_uni_rand_data_real(n_data, n_dims)
        similarity = lambda a, b: - ((a - b)**2).sum(axis=-1)  # negative Euclidean 2-norm
        query_data = gen_uni_rand_data_real(n_queries, n_dims)
        epsilon = 1
        query_data = gen_planted_rand_data_real(query_data, n_data, R=0.1, epsilon=0.1)
        plot_pairwise_dist(data)

    else:
        raise Exception

    new_idxs = lsh_container.store(data)
    n_data = data.shape[0]
    assert new_idxs == list(range(n_data))
    alphas = range(1, l)
    # TODO: Is this equivalent to just averaging over *all* alpha-neighborhoods in our container? Is there an easy way
    #   to collect all these neighborhoods?
    a_n_nbs = []  # mean neighborhood size per alpha
    a_sims = []  # mean similarity within a neighborhood per alpha

    # Look at each value for alpha
    for alpha in alphas:
        q_n_nbs = []  # neighborhood size for each query item
        q_sims = []  # mean intra-neighborhood similarity for each query item

        # Try each query item
        for q in query_data:
            nb_idxs = lsh_container.query(q, alpha=alpha)
            n_nbs = len(nb_idxs)
            mean_sim = np.mean([similarity(lsh_container.data[i], lsh_container.data[j]) for i in nb_idxs
                               for j in nb_idxs if i > j])
            q_n_nbs.append(n_nbs)
            q_sims.append(mean_sim)
            # print(nb_idxs)
            # print([1 - jaccard(lsh_container.data[q_idx], lsh_container.data[nb_idx]) for nb_idx in nb_idxs])
        a_n_nbs.append(np.mean(q_n_nbs))
        a_sims.append(np.mean(q_sims))

    scheme_dir = os.path.join(figs_dir, scheme)
    plt.plot(alphas, a_n_nbs)
    plt.ylabel('mean number of neighbors')
    plt.xlabel('alpha')
    plt.savefig(f'{scheme_dir}/mean_nbs_X_alpha')
    plt.close()

    # a_sims = np.where(np.isnan(a_sims), 1, a_sims)
    plt.plot(alphas, a_sims)
    plt.ylabel('mean intra-neighborhood similarity')
    plt.xlabel('alpha')
    plt.tight_layout()
    plt.savefig(f'{scheme_dir}/mean_sim_X_alpha')
    plt.close()

    # test_idx = 42
    # bucket_ids = lsh.data_idx_to_bucket_ids[test_idx]
    # collision_idxs = list(set(sum([lsh.tables[hash_idx][bucket_id] for hash_idx, bucket_id in enumerate(bucket_ids)], [])))
    # collision_idxs.remove(test_idx)
    # print(f"Test index {test_idx} collides with {collision_idxs}")
    # print("Average Jaccard similarity of collisions:", np.mean([jaccard(data[test_idx], data[collision_idx]) for collision_idx in collision_idxs]))


if __name__ == "__main__":
    plot_mean_nbs_by_alpha(scheme='pStable')
    plot_mean_nbs_by_alpha(scheme='MinHash')

    # Minhash collision probabilities
    ks = np.arange(1, 20)
    ls = np.arange(20, 21)
    plot_collision_prob(ks=ks, ls=ls)
    plot_collision_prob_alpha(k=1, l=20)

    # p-stable collision probabilities
    ks = np.arange(1, 20)
    ls = np.arange(11, 21)
    # rs = np.linspace(0, 4.01, 0.1)
    rs = [1, 2, 3, 4]
    plot_collision_prob_pstable(ks, ls, rs)
