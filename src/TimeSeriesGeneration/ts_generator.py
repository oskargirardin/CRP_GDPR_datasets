from sdv.sequential.par import PARSynthesizer
import torch


class TSGenerator:
    """
    Class to generate synthetic time series
    Methods currently available are 'PARSynthesizer'
    """

    def __init__(self, df, metadata, method='PAR', verbose=False, cuda=False):
        """
        :param df: the dataframe, for ParSynthesizer it should be in Long Format, see DataProcessor
        :param metadata: metadata for the df to train on
        :param method: the method with which to generate the data
        :param n_epochs: number of epochs for which the model should be trained
        :param n_samples: the number of samples to generate
        :param verbose: print training progress
        :param cuda: is a GPU available
        """
        self.df = df
        self.metadata = metadata
        self.method = method
        self.verbose = verbose
        self.cuda = torch.cuda.is_available()
        self.synthetic_df = None
        self.model = None


    def train(self, n_epochs = 100):
        if self.method == 'PAR':
            synthesizer = PARSynthesizer(self.metadata, verbose=self.verbose, cuda=self.cuda, epochs=n_epochs)
            synthesizer.fit(self.df)
            self.model = synthesizer
        else:
            print('The requested method has not been implemented')

    def sample(self, n_samples = 10, sequence_length = None):
        """
        Fits the synthesizer and samples synthetic data
        :return: returns the synthetic data in a dataframe that follows the metadata parameter
        """
        
        self.synthetic_df = self.model.sample(num_sequences=n_samples, sequence_length = sequence_length)
        return self.synthetic_df
        
