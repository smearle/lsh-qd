

class ExperimentRunner()
    '''
    This class contains the general logic for performing experiments on real or synthetic data using a provided lsh scheme.
    It takes in an lsh_instance (i.e. AlphaLSH, MultiProbeLSH, etc.) as well as a dataset (NOTE: we should come up with some unified
    way to handle different datasets, which all probably have different structures and data tyes -- perhaps they are instances of another
    class). It contains methods that will split the dataset into train and test splits, as well as collecting the ground-truth labels for
    near neighbors.

    Most of the methods, however, will be functions that run specific experiments -- i.e. evaluating the performance of a particular lsh
    scheme on a particular dataset
    '''
    def __init__(self):
        pass


    def evaluate_lsh(self, lsh_instance, dataset, query_data):
        '''
        Run the lsh scheme on all the points in the dataset, and then test it using the provided query pointd
        '''

    def 

'''
for data_point in test_set:
    container.query(data_point)
    add that to a list

compute statistics here



Container.compute_statistics():
    -collect the test set
    -iterate through it, call self.lsh_instance.query()

'''