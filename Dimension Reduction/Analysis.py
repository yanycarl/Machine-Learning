
from mylibs import ReadFile
from mylibs import Reduction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits.mplot3d


class Analysis:

    @staticmethod
    def show3D(m):
        ax = plt.subplot(projection='3d')
        ax.scatter(m[0], m[1], m[2])
        plt.show()

    @staticmethod
    def showPre():
        z1 = (i.matrix[0] - np.mean(i.matrix[0])) / np.std(i.matrix[0])
        z2 = (i.matrix[1] - np.mean(i.matrix[1])) / np.std(i.matrix[1])
        plt.scatter(z1, z2, marker='.', c=[0.5, 0.5, 0.5], s=5, edgecolors="None")
        sns.kdeplot(z1, z2, cmap="Reds")
        plt.plot(
            -3 * np.sqrt(ii.eigVal[1]) * np.array([0, ii.eigVec[1, 0]]),
            -3 * np.sqrt(ii.eigVal[1]) * np.array([0, ii.eigVec[1, 1]]),
            c=[1, 1, 1],
            linewidth=3
        )
        plt.plot(
            3 * np.sqrt(ii.eigVal[0]) * np.array([0, ii.eigVec[0, 0]]),
            3 * np.sqrt(ii.eigVal[0]) * np.array([0, ii.eigVec[0, 1]]),
            c=[1, 1, 1],
            linewidth=3
        )
        plt.plot(
            -3 * np.sqrt(ii.eigVal[1]) * np.array([0, ii.eigVec[1, 0]]),
            -3 * np.sqrt(ii.eigVal[1]) * np.array([0, ii.eigVec[1, 1]]),
            c=[0, 0.6, 0],
            linewidth=2
        )
        plt.plot(
            3 * np.sqrt(ii.eigVal[0]) * np.array([0, ii.eigVec[0, 0]]),
            3 * np.sqrt(ii.eigVal[0]) * np.array([0, ii.eigVec[0, 1]]),
            c=[0, 0.6, 0],
            linewidth=2
        )
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    @staticmethod
    def showPre2(m):
        plt.scatter(m[0].reshape(1985, 1), m[1].reshape(1985, 1))
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    @staticmethod
    def showNew(m):
        plt.scatter(np.transpose(m)[0], np.transpose(m)[1])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    @staticmethod
    def showNewKDE(m):
        sns.kdeplot(np.transpose(m)[0], np.transpose(m)[1])
        plt.scatter(np.transpose(m)[0], np.transpose(m)[1])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    @staticmethod
    def showVarCum(m):
        temp = [0]
        temp2 = 0.0
        for j in m:
            temp2 += j
            temp.append(temp2)
        temp /= temp2
        temp2 = np.arange(0, 9, 1)
        plt.plot(temp2, temp)
        print(temp)
        plt.xlabel("Remain Dimension")
        plt.ylabel("Information Keep Percent")
        plt.show()


if __name__ == '__main__':

    i = ReadFile.ReadFile()
    i.openFile()
    i.getSomeCols()
    ii = Reduction.Reduction()
    ii.centralization(i.matrix)
    ii.getCovMatrix()
    ii.getElg()
    ii.getSingular(i.matrix)
    # new = ii.decrease2(i.matrix)
    new = ii.decrease2(i.matrix)
    instance3 = Analysis()
    Analysis.showPre()
    Analysis.showPre2(ii.centeredM)
    Analysis.showNew(new)
    Analysis.show3D(i.matrix)
    Analysis.showNewKDE(new)
    Analysis.showVarCum(ii.eigVal)
