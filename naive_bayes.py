from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod

class NaiveBayes(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_training_files(self):
        pass
    
    def test(self):
        TEST_SPAM_PATH = "./enron2/spam"
        TEST_HAM_PATH = "./enron2/ham"
        TEST_SPAM_FILES = [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if isfile(join(TEST_SPAM_PATH, file))]
        TEST_HAM_FILES = [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if isfile(join(TEST_HAM_PATH, file))]
        correct = 0
        total = 0
        for ham_file in TEST_HAM_FILES:
            if not self.predict(ham_file):
                correct += 1
            total += 1
        for spam_file in TEST_SPAM_FILES:
            if self.predict(spam_file):
                correct += 1
            total += 1
        return float(correct)/total


    def predict(self, filename):
        pass