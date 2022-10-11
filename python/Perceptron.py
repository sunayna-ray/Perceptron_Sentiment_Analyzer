import sys
import getopt
import os
import math
import operator
import numpy as np
import random as rand

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds = 10
    self.vocab = set()
    self.vocabSize = 0
    self.training = []
    self.weights = None
    self.bias = 0
    self.idxZip = dict()

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """


    # Write code here
    phiN = np.zeros(self.vocabSize)
    for word in words:
        if word in self.vocab:
            idex = self.idxZip[word]
            phiN[idex] = phiN[idex] + 1

    out = self.bias + np.dot(self.weights,phiN)

    if out>=0:
        return 'pos'
    else:
        return 'neg'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """

    # Write code here
    phiN = np.zeros(self.vocabSize)
    for word in words:
      indx = self.idxZip[word]
      phiN[indx] = phiN[indx]+1

    self.training.append((phiN,klass))
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      """
      
      for example in split.train:
          words = example.words
          for w in set(words):
            self.vocab.add(w)

      self.vocabSize = len(self.vocab)
      # set index for each word for embedding
      self.idxZip = {word:idx for idx,word in enumerate(self.vocab)}

      for example in split.train:
          self.addExample(example.klass, example.words)

      wtAvg = np.zeros(self.vocabSize)
      biasAvg = 0

      wtCurr = np.zeros(self.vocabSize)
      biasCurr = 0

      vectCount = 1

      for iter in range(iterations):
        rand.shuffle(self.training)
        for phiN,y in self.training:
            if(y=='neg'): y=-1
            else: y = 1
            out = (biasCurr + np.dot(wtCurr,phiN))*y

            if out<=0:
              wtCurr += y*phiN
              wtAvg += vectCount * y * phiN
              biasCurr += y
              biasAvg += vectCount * y
            vectCount = vectCount+1

      self.bias = biasCurr-(biasAvg/vectCount) 
      self.weights= wtCurr - (wtAvg/vectCount)
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print('[INFO]\tAccuracy: %f' % accuracy)
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
# python Perceptron.py "../data/imdb1" <iteration_count>