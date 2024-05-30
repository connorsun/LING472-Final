from os import listdir, scandir
from os.path import isfile, join
import numpy as np
from gensim.models import Word2Vec

from naive_bayes import NaiveBayes
from nltk.tokenize import sent_tokenize

class NaiveBayesBasic(NaiveBayes):
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}
        self.word2vec_model = None
        self.similarity_map = {}

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
        spam_emails = []
        ham_emails = []
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            build = ""
            for line in file:
                line = line.replace("Subject:", " ")
                line = line.replace("\n", " ")
                tokens = line.strip().split()
                ham_total += len(tokens)
                for token in tokens:
                    ham_freq[token] = ham_freq.setdefault(token, 0) + 1
                build = build + " " + line
            file.close()
            build_array = sent_tokenize(build)
            build_array_array = []
            for token in build_array:
                build_array_array.append(token.split(" "))
            ham_emails.append(build_array_array)
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            build = ""
            for line in file:
                line = line.replace("Subject:", " ")
                line = line.replace("\n", " ")
                tokens = line.strip().split()
                spam_total += len(tokens)
                for token in tokens:
                    spam_freq[token] = spam_freq.setdefault(token, 0) + 1
                build = build + " " + line
            file.close()
            build_array = sent_tokenize(build)
            build_array_array = []
            for token in build_array:
                build_array_array.append(token.split(" "))
            spam_emails.append(build_array_array)
        self.ham_total = float(ham_total)
        self.ham_freq = ham_freq
        self.spam_total = float(spam_total)
        self.spam_freq = spam_freq
        self.word2vec_model = Word2Vec(sentences=[piece for word in (ham_emails + spam_emails) for piece in word],
                                       vector_size=100, window=5, min_count=1, workers=4)

    def predict(self, filename):
        file = open(filename, "r", encoding="ISO-8859-1")
        total_log_ham = np.log(self.ham_total / (self.ham_total + self.spam_total))
        total_log_spam = np.log(self.spam_total / (self.ham_total + self.spam_total))
        vector_list = []

        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                if token in self.word2vec_model.wv:
                    vector_list.append(token)
                total_log_ham += np.log((self.ham_freq.get(token, 0) + 1) / (self.ham_total + 2))
                total_log_spam += np.log((self.spam_freq.get(token, 0) + 1) / (self.spam_total + 2))

        if len(vector_list) < 10:
            return total_log_spam >= total_log_ham
        avg_vector = np.mean([self.word2vec_model.wv[vector] for vector in vector_list], axis=0)
        n = round(len(vector_list) / 10)
        similar_list = (self.word2vec_model.wv.most_similar(avg_vector, topn=n))
        for similar in similar_list:
            total_log_ham += similar[1] * np.log((self.ham_freq.get(similar[0], 0) + 1) / (self.ham_total + 2))
            total_log_spam += similar[1] * np.log((self.spam_freq.get(similar[0], 0) + 1) / (self.spam_total + 2))
        return total_log_spam >= total_log_ham


if __name__ == "__main__":
    nb = NaiveBayesBasic()
    nb.train()
    accuracy = nb.test()
    print(accuracy)