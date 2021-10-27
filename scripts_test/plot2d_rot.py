import numpy as np
import matplotlib.pyplot as plt
from glob import glob

categories = ['201703', '201803', '201806', '2018070', '201807[1-2]']
x_bins = np.linspace(-4.5,4.5,25)
y_bins = np.linspace(-0.5,11.5,13)

for category in categories:
    print(category)
    files1 = glob('/home/vbelissen/Downloads/train/' + category + '*_rot_tab.npy')
    files2 = glob('/home/vbelissen/Downloads/test_sync/' + category + '*_rot_tab.npy')
    files = files1 + files2
    N = len(files)
    print(N)
    valuesA = np.zeros(N*12)
    valuesB = np.zeros(N*12)
    valuesC = np.zeros((N,12))
    for i in range(N):
        tmp_npy = np.load(files[i])
        values = tmp_npy[:,-1]
        valuesA[12*i:12*(i+1)] = values
        valuesB[12*i:12*(i+1)] = np.arange(12)
        valuesC[i, :] = values
    valuesMean   = np.mean(valuesC,   axis=0)
    valuesStd    = np.std(valuesC,    axis=0)
    valuesMedian = np.median(valuesC, axis=0)
    print(valuesMean)
    print(valuesMedian)
    plt.hist2d(valuesA, valuesB, bins=[x_bins,y_bins], cmap=plt.cm.jet)
    plt.plot(valuesMean, np.arange(12),'.r-')
    #plt.plot(valuesMean - valuesStd, np.arange(12),'.m-')
    #plt.plot(valuesMean + valuesStd, np.arange(12),'.m-')
    plt.plot(valuesMedian, np.arange(12),'.g-')
    plt.colorbar()
    plt.show()

