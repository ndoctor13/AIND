import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selectedModel = None
        selectedBIC = float('inf')
        BIC = float('inf')
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                ##Train the feature set through Gaussian HMM
                model = GaussianHMM(n_components=num_hidden_states, covariance_type = "diag", n_iter=1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                ##Find the score of the trained dataset
                logL = model.score(self.X, self.lengths)

                p = num_hidden_states ** 2 + 2 * num_hidden_states * model.n_features - 1
                n = len(self.sequences)
                ##Test liklihood of the fit
                BIC = -2 * logL + (p * np.log(n))
            except:
                pass
            if BIC < selectedBIC:
                selectedBIC = BIC
                selectedModel = model
        return selectedModel ##returned the trained model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        selectedModel = None
        selectedDIC = float('-inf')
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_hidden_states, covariance_type = "diag", n_iter=1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                logL_thisword = model.score(self.X, self.lengths)
                ## Look for the log score of the trained data set to other words and see the fit. 
                avgLogL_otherwords = self.scoreOtherWords(model)
                ##If DIC score is lower that means the trained data set fits well with all words and thats not a good sign so ignore this data and updates states
                DIC = logL_thisword - avgLogL_otherwords
                if DIC > selectedDIC:
                    selectedBIC = DIC
                    selectedModel = model
            except:
                if selectedModel is None:
                    selectedModel =  GaussianHMM(n_components=num_hidden_states, covariance_type = "diag", n_iter=1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                pass
        return selectedModel

    def scoreOtherWords(self,model):
        wordCount = 0
        totalScore = 0
        for word,features in self.hwords.items():
            if self.this_word != word:
                X, lengths = features
                try:
                    score = model.score(X, lengths)
                    totalScore = score + totalScore
                    wordCount = wordCount + 1
                except:
                    pass
        average_score = totalScore / wordCount
        return average_score


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        n_split = min(len(self.lengths), 3)
        selectedModel = None
        selectedScore = float('-inf')
        
        logs = []
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                split_methods = KFold(n_split)
            except ValueError:
                ##If number of samples is one then no split of the data set can occur so train the set to HHM and return the model
                if len(self.sequences) == 1:
                    return GaussianHMM(n_components=num_hidden_states, covariance_type = "diag", n_iter=1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                else: 
                    split_methods = KFold(n_splits = 2)
            for cv_train_idx, cv_test_idx in split_methods.split(self.sequences): ## Split the dataset to training and test based on the splits
                try:
                    X, lengths = combine_sequences(cv_train_idx , self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    model = GaussianHMM(n_components=num_hidden_states, covariance_type = "diag", n_iter=1000, random_state = self.random_state, verbose = False).fit(X, lengths)
                    logL = model.score(X_test, lengths_test)
                    logs.append(logL)
                except:
                    pass
            average = np.mean(logs)
            if average > selectedScore:
                selectedModel = model
                selectedScore = average
        return selectedModel


