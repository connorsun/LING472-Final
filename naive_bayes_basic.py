from os import listdir
from os.path import isfile, join
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}

    def get_training_files(self):
        SPAM_PATH = "./enron1/spam"
        HAM_PATH = "./enron1/ham"
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        SPAM_FILES = [join(SPAM_PATH, file) for file in listdir(SPAM_PATH) if isfile(join(SPAM_PATH, file))]
        HAM_FILES = [join(HAM_PATH, file) for file in listdir(HAM_PATH) if isfile(join(HAM_PATH, file))]
        ham_total = 0
        ham_freq = {}
        spam_total = 0
        spam_freq = {}
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            for line in file:
                tokens = line.strip().split()
                ham_total += len(tokens)
                for token in tokens:
                    ham_freq[token] = ham_freq.setdefault(token, 0) + 1
            file.close()
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            for line in file:
                tokens = line.strip().split()
                spam_total += len(tokens)
                for token in tokens:
                    spam_freq[token] = spam_freq.setdefault(token, 0) + 1
            file.close()
        self.ham_total = float(ham_total)
        self.ham_freq = ham_freq
        self.spam_total = float(spam_total)
        self.spam_freq = spam_freq
    
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
        file = open(filename, "r", encoding="ISO-8859-1")
        total_log_ham = np.log(self.ham_total/(self.ham_total + self.spam_total))
        total_log_spam = np.log(self.spam_total/(self.ham_total + self.spam_total))
        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                total_log_ham += np.log((self.ham_freq.get(token, 0) + 1)/(self.ham_total + 2))
                total_log_spam += np.log((self.spam_freq.get(token, 0) + 1)/(self.spam_total + 2))
        return total_log_spam >= total_log_ham
        
if __name__ == "__main__":
    nb = NaiveBayes()
    nb.get_training_files()
    accuracy = nb.test()
    print(accuracy)