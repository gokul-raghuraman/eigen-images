import sys
import os
import warnings
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as Plot
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import tsne

def scaleImage(inputImg, width, height):
    scaledImg = cv2.resize(inputImg, (width, height)) 
    return scaledImg
    

def plotPoints3D(sampleDict):
    
    markers = ["o", "v", "s", "+", "x"]
    colors = ["blue", "red", "green", "cyan", "magenta", "yellow", "black", "white"]
    
    fig = plt.figure(figsize = (8,8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    
    for i, category in enumerate(sampleDict):
        sample = sampleDict[category]
        marker = markers[i]
        color = colors[i]
        ax.plot(sample[0,:], sample[1,:], sample[2,:], marker, markersize=8, color=color, alpha=0.5, label=category)
    
    plt.title('Plot after dimensionality reduction')
    ax.legend(loc='upper right')
    plt.show()


def plotPoints2D(sampleDict):
    
    markers = ["o", "v", "s", "+", "x"]
    colors = ["blue", "red", "green", "cyan", "magenta", "yellow", "black", "white"]
    
    for i, category in enumerate(sampleDict):
        sample = sampleDict[category]
        marker = markers[i]
        color = colors[i]
        plt.plot(sample[0,:], sample[1,:], marker, markersize=8, color=color, alpha=0.5, label=category)
    
    plt.xlim([-3000,3000])
    plt.ylim([-3000,3000])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples for apple')
    plt.show()


def showHist(inputArray):
    """
    Plots a histogram of intensity levels vs occurrence
    
    :Parameters:
        inputArray : `numpy.ndarray`
            array of intensity levels
    :Returns:
        None
    """
    hist = getHist(inputArray)
    plt.bar(range(0, 256), [hist[key] for key in hist], 5, 1)
    plt.show()
    
    
def getHist(input):
    """
    Returns a histogram dict of intensity level vs occurrence
    
    :Parameters:
        input : `numpy.ndarray`
            array of intensity levels
    :Returns:
        The histogram
    :Rtype:
        `dict`
    """
    hist = {}
    brightnessRange = range(0, 256)
    
    for val in brightnessRange:
        hist[val] = 0
    inputList = input.tolist()
    
    for val in hist:
        hist[val] = inputList.count(val) 
    return hist


def saveDataMatrix(dataMatrix, filePath='./data/raw/pictionary_data_matrix.npy'):
    """
    Saves the given data matrix to disk.
    
    :Parameters:
        dataMatrix : `numpy.ndarray`
            The data matrix to save
        filePath : `str`
            The file path to save at
    :Returns:
        None
    """
    if not (filePath.startswith('/') or filePath.startswith('./')):
        filePath = filePath.split('/')[-1]
        if not os.path.isdir('./data/raw'):
            os.makedirs('./data/raw')
        filePath = './data/raw/%s' % filePath
    if not filePath.endswith('.npy'):
        filePath = '%s.npy' % filePath
    np.save(filePath, dataMatrix)
    print('\n\tSaved data to %s.' % filePath)


def loadDataMatrix(filePath='./data/raw/pictionary_data_matrix.npy'):
    """
    Loads and returns a single data matrix from the file path specified
    
    :Paramaters: 
        filePath : `str`
            The file path to read
    :Returns:
        The data matrix
    :Rtype:
        `numpy.ndarray`
    """
    if not (filePath.startswith('/') or filePath.startswith('./')):
        filePath = filePath.split('/')[-1]
        filePath = './data/raw/%s' % filePath
    if not filePath.endswith('.npy'):
        filePath = '%s.npy' % filePath
    return np.load(filePath)


def buildDataMatrixFromDisk(baseDir='./data/raw', category='airplane'):
    """
    Builds a data matrix from file for the specified category.
    
    :Parameters:
        baseDir : `str`
            The base directory to look for data
        category : `str`
            The category for which to build the matrix
            
    :Returns:
        The combined data matrix
    :Rtype:
        `numpy.ndarray`
    """
    mFile = '%s.npy' % category
    dataMatrix = loadDataMatrix(mFile)
    return dataMatrix
    

def getVectorFromImage(filePath):
    """
    Method that reads an image file and returns a linear array of size M * N
    Where M and N are original dimensions of the image
    
    :Parameters:
        filePath : `str`
            The path to the file
    :Returns:
        The 1 X N vector of image intensities (N is the total number of pixels)
    :Rtype:
        `numpy.ndarray`
    """
    img = cv2.imread(filePath)
    
    #Scaling image to 200 x 200
    img = scaleImage(img, 200, 200)
    return getVectorFromMatrix(img[:,:,0])


def getMatrixFromVector(inputVector):
    width = math.sqrt(len(inputVector))
    return np.reshape(inputVector, (width, width))


def getVectorFromMatrix(matrix):
    return matrix.flatten()
    

def getDataMatrix(baseDir='./images', category='airplane', force=False):
    """
    Crawls the images directory and builds an N X K intensity data matrix where 
        K = number of images
        N = number of pixels per image 
    This method generates and saves the category's data matrix into a data 
    directory to memory once built. During each call to build, if the data matrix 
    exists on disk already, then don't rebuild if force flag is set to False.
    
    :Parameters:
        baseDir : `str`
            The base image directory path. Defaults to './images'.
        category : `str`
            The category for which to get the matrix
        force : `bool`
            If True, rebuild each image category's data from scratch. 
            Else reuse each image category's data matrix from disk.
    :Returns:
        N X K data matrix
    :Rtype:
        `numpy.ndarray`
    """
    if os.path.exists('./data/%s.npy' % category) and not force:
        print('Data for %s exists on disk. Will reuse.' % category)
        dataMatrix = buildDataMatrixFromDisk(category=category)
        return dataMatrix
            
    imgDir = '%s/%s' % (baseDir, category)
    print('\nProcessing directory /%s' % category)
    numFiles = 1
    fileNames = [fileName for fileName in os.listdir(imgDir) if (fileName.endswith('png') \
                 or fileName.endswith('.jpg'))]
    totalNumFiles = len(fileNames)
    dataMatrix = np.array([])
    for fileName in fileNames:
        filePath = "%s/%s" % (imgDir, fileName)
        imgVec = getVectorFromImage(filePath)
        if (len(dataMatrix) == 0):
            dataMatrix = imgVec
        else:
            dataMatrix = np.vstack((dataMatrix, imgVec))
        sys.stdout.write('\r\tImages Processed : %s / %s' % (numFiles, totalNumFiles))
        sys.stdout.flush()
        numFiles += 1

    saveDataMatrix(dataMatrix, filePath=category)
    return dataMatrix
         
            
def getAvgVector(dataMatrix):
    avgVector = np.zeros(dataMatrix.shape[1])
    for i in range(len(avgVector)):
        avgVector[i] = np.average(dataMatrix[:,i])
    return avgVector
    

def invertImage(imageMatrix):
    for i in range(imageMatrix.shape[0]):
        for j in range(imageMatrix.shape[1]):
            imageMatrix[i][j] = 255 - imageMatrix[i][j]
    return imageMatrix 


def getDiffMatrix(dataMatrix, avgVector):
    diffMatrix = np.array([])
    for i in range(dataMatrix.shape[0]):
        diffVec = np.subtract(dataMatrix[i, :], avgVector)
        if len(diffMatrix) == 0:
            diffMatrix = diffVec
        else:
            diffMatrix = np.vstack((diffMatrix, diffVec))
    return diffMatrix
    

def getEigenImages_backup(dataMatrix, avgVector, diffMatrix):
    covMatrix = np.dot(diffMatrix, diffMatrix.T)
    eigenImages = []
    for i in range(dataMatrix.shape[0]):
        eigenVec = np.dot(diffMatrix.T, covMatrix[i, :])
        eigenImage = getMatrixFromVector(eigenVec)
        
        #inverting the image
        eigenImage = invertImage(eigenImage)
        
        eigenImages.append(eigenImage)
    return eigenImages
    

def getEigenImages(dataMatrix, avgVector, diffMatrix):
    smallCovMatrix = np.dot(diffMatrix, diffMatrix.T)
    
    eigenImages = []
    for i in range(smallCovMatrix.shape[0]):
        smallEigenVec = smallCovMatrix[i]
        largeEigenVec = np.dot(smallEigenVec, diffMatrix)
        eigenImage = getMatrixFromVector(largeEigenVec)
        #inverting the image
        eigenImage = invertImage(eigenImage)
        eigenImages.append(eigenImage)
    return eigenImages


def loadEigenImages(baseDir='./eigen_images', category='airplane'):
    eigenImages = []
    imgDir = '%s/%s' % (baseDir, category)
    fileNames = [fileName for fileName in os.listdir(imgDir) if (fileName.endswith('png') \
                 or fileName.endswith('.jpg'))]
    for fileName in fileNames:
        filePath = '%s/%s' % (imgDir, fileName)
        eigenImage = getVectorFromMatrix(cv2.imread(filePath)[:,:,0])
        eigenImages.append(eigenImage)
    
    return eigenImages


def loadWeights(baseDir='./eigen_weights', category='airplane'):
    filePath = '%s/%s_weights.npy' % (baseDir, category)
    return np.load(filePath)
    

def saveImages(images, baseDir='./eigen_images', category='airplane'):
    filePath = '%s/%s' % (baseDir, category)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    print('\n')
    for i in range(len(images)):
        fileName = '%s/%s.png' % (filePath, i + 1)
        cv2.imwrite(fileName, images[i])
        print('Saved image : %s' % (fileName))
    
    
def saveWeights(weights, baseDir='./eigen_weights', category='airplane'):
    if not os.path.isdir(baseDir):
        os.makedirs(baseDir)
    filePath = '%s/%s_weights' % (baseDir, category)
    print("SAVING WEIGHTS : " + str(weights))
    print("...INTO " + str(filePath))
    np.save(filePath, weights)


def buildAndSaveEigenImages(dataMatrix, category, avgVector, diffMatrix):
    eigenImages = getEigenImages(dataMatrix, avgVector, diffMatrix)
    saveImages(eigenImages, baseDir='./eigen_images', category=category)
    saveImages([getMatrixFromVector(avgVector)], baseDir='./average_images', category=category)
    weights = getWeights(dataMatrix, eigenImages, diffMatrix)
    saveWeights(weights, baseDir='./eigen_weights', category=category)


def getWeights(dataMatrix, eigenImages, diffMatrix):
    avgVector = getAvgVector(dataMatrix)
    eigenImageVector = getVectorFromMatrix(eigenImages[0])
    weights = np.array([])
    for i in range(len(eigenImages)):
        eigenImageVector = getVectorFromMatrix(eigenImages[i])
        diffVector = diffMatrix[i]
        weight = np.dot(diffVector, eigenImageVector)
        weights = np.concatenate((weights, [weight]))
    return weights
    
    
def getProbeWeights(eigenImages, diffVector):
    weights = np.array([])
    for eigenImage in eigenImages:
        weight = np.dot(eigenImage, diffVector)
        weights = np.concatenate((weights, [weight]))
    return weights

def buildData(categories, baseDir='./images'):
    for category in categories:
        dataMatrix = getDataMatrix(baseDir=baseDir, category=category, force=False)
        avgVector = getAvgVector(dataMatrix)    
        diffMatrix = getDiffMatrix(dataMatrix, avgVector)
        buildAndSaveEigenImages(dataMatrix, category, avgVector, diffMatrix)


def runPCA(dataMatrix, reducedDims=50):
    """
    Runs PCA for the given data matrix
    
    :Parameters:
        dataMatrix : `numpy.ndarray`
            The input data
        reducedDims : `int`
            The reduced number of dimensions
    :Returns: 
        None
    """
    print('\n')
    print('*' * 49)
    print('***Running Pricipal Component Analysis on data***')
    print('*' * 49)
    print('\nNumber of samples             : %s' % dataMatrix.shape[0])
    print('\nOriginal number of dimensions : %s' % dataMatrix.shape[1])
    print('\nReduced number of dimensions  : %s' % reducedDims)
    pcaObj = PCA(n_components=reducedDims)
    pcaTransformedMat = pcaObj.fit_transform(dataMatrix)
    return pcaTransformedMat

def getEigenVecsFromDisk(baseDir='./pca_2', category='airplane'):
    filePath = baseDir + '/' + category + '_data.npy'
    return np.load(filePath)
    


"""
Run step
"""
baseDir = './images'
categories = [categoryDir for categoryDir in os.listdir(baseDir) if (not os.path.isfile("%s/%s" % (baseDir, categoryDir)))]

for category in categories:
    dataMatrix = getDataMatrix(baseDir=baseDir, category=category, force=True)
    avgVector = getAvgVector(dataMatrix)   
    diffMatrix = getDiffMatrix(dataMatrix, avgVector)
    buildAndSaveEigenImages(dataMatrix, category, avgVector, diffMatrix)


"""
Build and Save data
"""
#buildData(categories, baseDir=baseDir)


#Plotting 2-dimensional data
"""
for category in categories:
    eigenVec = getEigenVecsFromDisk(baseDir='./pca_2', category=category)
    print(eigenVec.shape)
    plotPoints2D({category : eigenVec})
   
    plt.plot(eigenVec[0,:], eigenVec[1,:], "o", markersize=8, color="blue", alpha=0.5, label=category)
    plt.xlim([-3000,3000])
    plt.ylim([-3000,3000])
    plt.show()
   
    
"""
    
#For getting weights
"""
for category in categories:
    dataMatrix = getDataMatrix(baseDir=baseDir, category=category, force=False)
    avgVector = getAvgVector(dataMatrix)   

    #Get diff vector
    diffVector = np.subtract(imgVector, avgVector)
    
    #Load eigen images from disk
    eigenImages = loadEigenImages(baseDir='./eigen_images', category=category)

    #Find category weights
    weights = loadWeights(baseDir='./eigen_weights', category=category)
"""

#For saving 2-dimensional PCA data
"""
for category in categories:
    dataMatrix = getDataMatrix(baseDir=baseDir, category=category, force=True)
    trans = runPCA(dataMatrix, reducedDims=3)
    
    print('Saving ' + category)
    np.save('./pca_2/' + category + '_data.npy', trans.T)
    
    
    catDict[category] = trans.T
"""

    
    
