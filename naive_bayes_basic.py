from os import listdir, scandir
from os.path import isfile, join
import numpy as np
from naive_bayes import NaiveBayes

class NaiveBayesBasic(NaiveBayes):
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}

    def train(self):
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        SPAM_FILES = []
        HAM_FILES = []
        TESTING_PATH = "./Training"
        TESTING_FOLDERS_PATHS = [f.path for f in scandir(TESTING_PATH) if f.is_dir()]
        for testing_folder_path in TESTING_FOLDERS_PATHS:
            TEST_SPAM_PATH = join(testing_folder_path, "spam")
            TEST_HAM_PATH = join(testing_folder_path, "ham")
            SPAM_FILES += [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if isfile(join(TEST_SPAM_PATH, file))]
            HAM_FILES += [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if isfile(join(TEST_HAM_PATH, file))]
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
    nb = NaiveBayesBasic()
    nb.train()
    accuracy = nb.test()
    print(accuracy)