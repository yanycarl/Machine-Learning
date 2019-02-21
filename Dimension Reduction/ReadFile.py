
import csv
import numpy as np


class ReadFile:

    def __init__(self):
        self.matrix = []

    def openFile(self):
        with open('D:\\DATA\\ml\\female.csv', newline='\n') as csvFile:
            read = csv.reader(csvFile, delimiter=',', quotechar='|')
            i = 0
            for row in read:
                i += 1
                if i > 2:
                    temp = []
                    for item in row:
                        temp.append(item)
                    self.matrix.append(temp)
        self.matrix = np.transpose(self.matrix)

    def getSomeCols(self):
        self.matrix = self.matrix[1:9]
        self.matrix = self.matrix.reshape(8, 1985)
        self.matrix = self.matrix.astype(float)
