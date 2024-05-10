import time
from naive_bayes import NaiveBayes
from ngram_bayes import NgramBayes

class CompositeBayes(NaiveBayes):
    def __init__(self, bayeses):
        self.bayeses = bayeses

    def train(self):
        for bayes in self.bayeses:
            bayes.train()

    def predict(self, filename):
        poll = 0
        for bayes in self.bayeses:
            poll += 1 if bayes.predict(filename) else -1
        return poll >= 0

def create_composite_ngram(max_n):
    return [NgramBayes(i + 1) for i in range(max_n)]

if __name__ == "__main__":
    ts = time.time()
    for n in range(1, 6):
        nb = CompositeBayes(create_composite_ngram(n))
        nb.train()
        accuracy = nb.test()
        print("accuracy for n = " + str(n) + " " + str(accuracy))
    print("time taken " + str(time.time() - ts))
