import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for wordIndex in range(0, len(test_set.get_all_Xlengths())):
      X, length = test_set.get_all_Xlengths()[wordIndex] ##Get the trained data values and the length of the dataset. Use that to get the probablity score
      probability_dic = {}
      for word, model in models.items():
        try:
          logL = model.score(X,length)
          ##Add the probability score to the word
          probability_dic[word] = logL
        except:
          probability_dic[word] = float('-inf')
      probabilities.append(probability_dic)
      ##Find the best guess word the test word and store it.
      guess = max([(v,k) for k,v in probability_dic.items()])[1]
      guesses.append(guess)
    return (probabilities,guesses)
