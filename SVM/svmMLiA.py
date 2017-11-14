def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArray = line.strip().split('\t')
        dataMat.append([float(lineArray[0]), float(lineArray[1])])
        labelMat.append(float(lineArray[2]))
    return dataMat, labelMat

dataMat, labelMat = loadDataSet('testSet.txt')
