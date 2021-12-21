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


class LSHContainer():
    '''A data structure that uses (alpha-tunable) LSH to compute approximate nearest neighbors.'''
    def __init__(self,
                 k=1,
                 l=5,
                 lsh_cls=AlphaLSH,
                 lsh_args={},
                 seed=42):
        self.lsh = lsh_cls(k=k, l=l, seed=seed, **lsh_args)
        self.data = defaultdict(int)

    def hash(self, input_data):
        '''
        Store a batch of input data in the LSH scheme. The LSH scheme only stores data indices, which this container
        stores the actual datapoints.
        Args:
            input_data: a (n_data, n_dims)-shape array of new datapoints to be stored in the container
        '''
        input_data_idxs = np.empty(shape=input_data.shape[0], dtype=int)

        for row_idx, data_point in tqdm(enumerate(input_data), desc="Hashing data", total=input_data.shape[0]):
            data_idx, bucket_ids = self.lsh.hash(data_point)
            assert data_idx not in self.data
            self.data[data_idx] = data_point
            input_data_idxs[row_idx] = data_idx

        return input_data_idxs

    def query_idx(self, data_idx, alpha=1, threshold=0):
        return self.lsh.query_idx(data_idx, alpha=alpha)

    def query(self, x, **kwargs):
        return self.lsh.query(x, **kwargs)


class RankedNeighborContainer(object):
    '''
    A container which preprocess datasets to be able to efficiently identify approximate nearest neighbors for multiple
    distance thresholds (i.e. within balls of varying radii around each query point), by using a
    differently-tuned vanilla LSH scheme for each ball.
    '''
    def __init__(self, inner_radii, container_args):
        neighb_params = get_ranked_neighb_params(inner_radii, get_min_k_l_minhash)
        for k, l in neighb_params:
            plot_collision_prob(ks=[k], ls=[l])
        self.neighb_containers = [LSHContainer(*n_params, **container_args) for n_params in neighb_params]

    def hash(self, data):
        return [container.hash(data) for container in self.neighb_containers]

    def query(self, x):
        '''
        Returned a set of approximate neighbors for each neighborhood of interest.
        Args:
            x: query item
        Returns:
            neighbs: iterable of iterables; set of items within each neighborhood
        '''
        return [container.query(x) for container in self.neighb_containers]


class AlphaRankedNeighborContainer(object):
    '''
    A container which preprocess datasets to be able to efficiently identify approximate nearest neighbors for multiple
    distance thresholds (i.e. within balls of varying radii around each query point), by using a single alpha-tunable
    LSH scheme.
    '''
    def __init__(self, inner_radii, container_args):
        neighb_params = get_ranked_neighb_params_alpha(inner_radii, get_k_l_alpha_minhash)
        k, l, alphas = neighb_params
        self.alphas = alphas
        self.neighb_container = LSHContainer(k=k, l=l, **container_args)

    def hash(self, data):
        return self.neighb_container.hash(data)

    def query(self, x):
        '''
        Returned a set of approximate neighbors for each neighborhood of interest.
        Args:
            x: query item
        Returns:
            neighbs: iterable of iterables; set of items within each neighborhood
        '''
        return [self.neighb_container.query(x, alpha=a) for a in self.alphas]


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
    p_coll = np.zeros(sim.shape)

    for i in range(int(alpha), l + 1):
        p_coll_i = math.comb(l, i) * (sim ** k) ** i * (1 - sim ** k) ** (l - i)
        p_coll_i = p_coll_i.astype(np.float64)
        p_coll += p_coll_i

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

    assert len(ls) == 1

    if len(ks) > 1:
        alpha_fn = lambda ki: 0.2 + 0.8 * (ki / (len(ks) - 1))
        title = f'l-{ls[0]}'
    else:
        alpha_fn = lambda ki: 1
        title = f'k-{ks[0]}_l-{ls[0]}'

    for l in ls:
        l_cps = []

        for ki, k in enumerate(ks):
            cp = collision_prob(sim, k, l)
            l_cps.append(cp)
            label = f'k = {int(k)}' if ki == 0 or ki == len(ks) - 1 else None
            plt.plot(sim, cp, label=label, color='red', alpha=alpha_fn(ki))

        cps.append(l_cps)
        plt.legend()
        plt.xlabel('item similarity')
        plt.ylabel('collision probability')
        plt.savefig(os.path.join(f'{coll_probs_dir}', title))
        plt.close()


def plot_collision_prob_alpha(k, l):
    '''
    Plots the collision probability by distance of a tunable LSH scheme with given parameters.
    Args:
        r: number of bands in each LSH instance
        t: number of LSH instances
    '''
    v = np.arange(0, 1.01, 1 / 100)  # the
    alphas = np.arange(1, l + 1, 1)

    if len(alphas) > 1:
        alpha_fn = lambda ai: 0.2 + 0.8 * (ai / (len(alphas) - 1))
    else:
        alpha_fn = lambda ki: 1

    for ai, a in enumerate(alphas):
        p_coll = collision_prob_alpha(v, k, l, alpha=a)
        label = f'alpha = {int(a)}' if ai == 0 or ai == len(alphas) else None
        plt.plot(v, p_coll, label=label, color='blue', alpha=alpha_fn(ai))

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
        lsh_container = LSHContainer(l=l, k=k)
        data = gen_uni_rand_data_bin(n_data, n_dims)
        similarity = lambda a, b: 1 - jaccard(a, b)
        query_data = gen_uni_rand_data_bin(n_queries, n_dims)

    elif scheme == 'pStable':
        lsh_args = {
            'lsh_cls': pStableHash,
            'n_dims': n_dims}
        lsh_container = LSHContainer(l=l, k=k, lsh_cls=AlphaLSH, lsh_args=lsh_args)
        data = gen_uni_rand_data_real(n_data, n_dims)
        similarity = lambda a, b: - ((a - b)**2).sum(axis=-1)  # negative Euclidean 2-norm
        query_data = gen_uni_rand_data_real(n_queries, n_dims)
        epsilon = 1
        query_data = gen_planted_rand_data_real(query_data, n_data, R=0.1, epsilon=0.1)
        plot_pairwise_dist(data)

    else:
        raise Exception

    new_idxs = lsh_container.hash(data)
    n_data = data.shape[0]
    assert np.all(new_idxs == np.array(range(n_data)))
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


def get_k_l_alpha_minhash(posi_sims, false_sims, posi_rate=.90, false_rate=.10):

    # We know that when alpha = 1 (most permissive) we'll need to satisfy the most permissive threshold (largest ball)
    # in the vanilla setting.
    assert posi_sims[-1] == np.min(posi_sims) and false_sims[-1] == np.min(false_sims)
    posi_sim, false_sim = posi_sims[-1], false_sims[-1]
    valid_params = get_k_l_minhash(posi_sim, false_sim, posi_rate=posi_rate, false_rate=false_rate)
    all_thresh_satisfied = False

    # Keep increasing the "resolution" of our LSH scheme until we find one that is fine-grained enough
    for k, l in valid_params:
        alphas = []

        # Look for an alpha to satisfy each neighborhood ball
        for posi_sim, false_sim in zip(posi_sims, false_sims):
            alpha = get_alpha_minhash(k, l, posi_sim, false_sim, posi_rate=posi_rate, false_rate=false_rate)
            if alpha is None: break
            alphas.append(alpha)

        # Stop searching if we've found alphas to satisfy each threshold
        if len(alphas) == len(posi_sims): break
        else: raise Exception('Failed to find (k, l) allowing for alpha-tuned neighborhoods of specified sizes.')

    return k, l, alphas


def get_alpha_minhash(k, l, posi_sim, false_sim, posi_rate=0.9, false_rate=0.1):
    '''
    For fixed k and l, find an alpha setting that will satisfy the given thresholds, if one exists.
    Args:
        k:
        l:
        posi_sim:
        false_sim:
        posi_rate:
        false_rate:
    Returns:
    '''
    plot_collision_prob_alpha(k, l)
    # We're interested in alphas from 1 to l (total number of tables)
    alphas = np.arange(1, l + 1)
    p_ts = np.empty(alphas.shape)
    p_fs = np.empty(alphas.shape)
    for ai, alpha in enumerate(alphas):
        p_t, p_f = collision_prob_alpha(np.array([posi_sim, false_sim]), k, l, alpha=alpha)
        p_ts[ai] = p_t
        p_fs[ai] = p_f

    # We take the greatest (i.e. least permissive) valid alpha.
    # Note: we could also take the most permissive? Not sure what's best here.
    valid_alphas = np.argwhere((p_ts >= posi_rate) & (p_fs <= false_rate)) + 1
    alpha = valid_alphas[-1]

    return alpha if len(alpha) > 0 else None


def get_min_k_l_minhash(posi_sim, false_sim, posi_rate=.90, false_rate=.10):
    '''
    Get valid MinHash LSH parameters with least space complexity.
    Args:
        posi_sim:
        false_sim:
        posi_rate:
        false_rate:
    Returns:
        k: minimum number of bands required
        l: minimum number of tables  required
    '''
    valid_params = get_k_l_minhash(posi_sim, false_sim, posi_rate=posi_rate, false_rate=false_rate)
    k, l = valid_params[0]
    assert k == valid_params[:, 0].min() and l == valid_params[:, 1].min()

    return k, l


def get_k_l_minhash(posi_sim, false_sim, posi_rate=.90, false_rate=.10):
    '''
    Get LSH parameters to ensure a lower bound on false positives and upper bound on false negatives for items within
    a given similarity threshold
    Args:
        posi_sim: similarity threshold to be considered a neighbor
        false_sim: similarity threshold beyond which we want to guarantee low number of collisions
        posi_rate: lower bound of probability of successfully identifying a true neighbor
        false_rate: upper bound on probability of mistakenly identifying a false neighbor
    Returns:
    '''
    n_cell = 1000  # The range of values [1, n_cell+1] to check for r and t

    # SimHash parameter search
    # k is number of bands -- size of compressed representation produced by an LSH instance
    # l is the number of LSH instances
    k_idxs = np.arange(n_cell//2) + 1
    l_idxs = np.arange(n_cell) + 1

    # Note that k varies over columns, l varies between rows
    l, k = np.meshgrid(k_idxs, l_idxs)

    # Calculate chance of catching a true positive (false negative) above (below) a certain similarity threshold
    # These are both (n_cell, n_cell)-shape grids representing probabilities over different parameters
    p_tp = collision_prob(posi_sim, k, l)
    p_fp = collision_prob(false_sim, k, l)

    # We take the least parameters over these two dimensions (and assume them to also be the least of each).
    valid_params = np.argwhere((p_tp >= posi_rate) & (p_fp <= false_rate)) + 1

    return valid_params


def get_ranked_neighb_params(inner_radii, param_fn):

    # each near neighborhood is a ball, each anti-neighborhood is the complement of an err_width-larger ball
    err_width = .2
    p_sims = 1 - inner_radii
    f_sims = np.array(p_sims) - err_width
    neighb_params = [param_fn(posi_sim=p_sim, false_sim=f_sim) for p_sim, f_sim in zip(p_sims, f_sims)]

    return neighb_params


def get_ranked_neighb_params_alpha(inner_radii, param_fn):
    err_width = .2
    p_sims = 1 - inner_radii
    f_sims = np.array(p_sims) - err_width
    neighb_params = param_fn(p_sims, f_sims)

    return neighb_params


def plot_neighb_size(neighbs, xs, xlabel, title):
    # neighbs has shape (n_queries, n_neighborhoods)
    sizes = []
    for q_neighbs in neighbs:
        sizes.append([len(nb_idxs) for nb_idxs in q_neighbs])

    sizes = np.array(sizes)
    mean_sizes = sizes.mean(axis=0)
    plt.plot(xs, mean_sizes)
    plt.xlabel(xlabel)
    plt.ylabel('neighborhood size')
    plt.savefig(os.path.join(f'{coll_probs_dir}', f'n_nbs_X_radii {title}'))
    plt.close()

    return mean_sizes


def plot_neighb_sim(neighbs, data, xs, xlabel, title, sim_fn=jaccard):
    sims = []
    for q_neighbs in neighbs:
        sims.append([np.mean([sim_fn(data[xi], data[yi]) for xi in nb_idxs for yi in nb_idxs if xi != yi])
                              for nb_idxs in q_neighbs])
    sims = np.array(sims)
    mean_sims = np.nanmean(sims, axis=0)
    plt.plot(xs, mean_sims)
    plt.xlabel(xlabel)
    plt.ylabel('intra-neighborhood distance')
    plt.savefig(os.path.join(f'{coll_probs_dir}', f'nb_sim_X_radii {title}'))
    plt.close()

    return sims


def get_ranked_neighbs(scheme='MinHash'):
    '''
    Find ranked nearest-neighbors using both the alpha-tunable LSH container, and multiple vanilla LSH containers.
    Args:
        scheme (str): the underlying LSH scheme. So far, either 'MinHash' or 'pStable'
    '''
    radii = np.linspace(0.1, 0.2, 3)
    container = RankedNeighborContainer(radii,
                                        container_args={
                                            'lsh_cls': MinHash,
                                        })
    alpha_container = AlphaRankedNeighborContainer(radii,
                                                   container_args={
                                                       'lsh_cls': AlphaLSH,
                                                       'lsh_args': {
                                                           'lsh_cls': MinHash,
                                                       }
                                                   })
    dim = 100
    n_data = 1000
    n_query = 10
    data = gen_uni_rand_data_bin(n_data, dim)
    query_data = gen_uni_rand_data_bin(n_query, dim)
    data_idxs_a = alpha_container.hash(data)
    data_idxs = container.hash(data)
    assert np.all(data_idxs == data_idxs_a)
    neighbs = []
    neighbs_alpha = []

    # Get the neighborhoods surrounding each query point from the container
    for q in query_data:
        q_neighbs = container.query(q)
        q_neighbs_alpha = container.query(q)
        neighbs.append(q_neighbs)
        neighbs_alpha.append(q_neighbs_alpha)

    plot_neighb_size(neighbs, xs=radii, xlabel='radii', title=f'{scheme}')
    plot_neighb_sim(neighbs, data, xs=radii, xlabel='radii', title=f'{scheme}')
    plot_neighb_size(neighbs_alpha, xs=radii, xlabel='radii', title=f'Alpha{scheme}')
    plot_neighb_sim(neighbs_alpha, data, xs=radii, xlabel='radii', title=f'Alpha{scheme}')


def main():
    get_ranked_neighbs(scheme='MinHash')
    # TODO: implement ranked neighborhood experiment based on p-Stable LSH
    # get_ranked_neighbs(scheme='pStable')

    # Look at effect of alpha on neighborhood size/similarity, fixing other parameters
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


if __name__ == '__main__':
    main()
