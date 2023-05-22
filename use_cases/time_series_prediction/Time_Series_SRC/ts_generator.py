from sdv.sequential.par import PARSynthesizer
import torch


class TSGenerator:
    """
    Class to generate synthetic time series
    Methods currently available are 'PARSynthesizer'
    """

    def __init__(self, df, metadata, method='PAR', n_epochs=10, n_samples=10, verbose=False, cuda=False):
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
        self.n_epochs = n_epochs
        self.method = method
        self.n_samples = n_samples
        self.verbose = verbose
        self.cuda = torch.cuda.is_available()
        self.synthetic_df = None

    def generate(self):
        """
        Fits the synthesizer and samples synthetic data
        :return: returns the synthetic data in a dataframe that follows the metadata parameter
        """
        if self.method == 'PAR':
            synthesizer = PARSynthesizer(self.metadata, verbose=self.verbose, cuda=self.cuda, epochs=self.n_epochs)
            synthesizer.fit(self.df)
            self.synthetic_df = synthesizer.sample(num_sequences=self.n_samples)
            return self.synthetic_df
        else:
            print('The requested method has not been implemented')
