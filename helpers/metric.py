class Metric:
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
    def __repr__(self):
        return "precision {0}, recall {1}, accuracy {2}".format(self.precision,self.recall,self.accuracy)


    def add_true_positive(self):
        self.true_positives+=1

    def add_true_negative(self):
        self.true_negatives+=1
    def add_false_positive(self):
        self.false_positives+=1
    def add_false_negative(self):
        self.false_negatives+=1
    @property
    def precision(self):
        if self.true_positives + self.false_positives == 0:
            return 0
        return  (self.true_positives/(self.true_positives+self.false_positives)) * 100

    @property
    def recall(self):
        if self.true_positives + self.false_negatives == 0:
            return 0
        return (self.true_positives / (self.true_positives + self.false_negatives)) * 100
    @property
    def accuracy(self):
        if self.true_positives + self.true_negatives + self.false_positives + self.true_negatives == 0:
            return 0
        return 100*(self.true_positives + self.true_negatives)/(self.true_positives + self.true_negatives + self.false_positives + self.true_negatives)
