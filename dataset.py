# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
import random
import matplotlib.pyplot as plt

# global params
fld = 'data/'

sateliteImages = ['20090526', '20110514', '20120524', '20130608',
                  '20140517', '20150507', '20160526']
alt = 'DEM_altitude.tif'
slp = 'DEM_slope.tif'


def loadSateliteFile(date, normalize=False):
    img = io.imread(fld + date + ".tif").astype(np.float32)
    ndvi = io.imread(fld + date + "_NDVI.tif").astype(np.float32)
    mask = io.imread(fld + date + "_mask_ls.tif").astype(np.float32)
    if normalize:
        img /= 20000.0
        ndvi /= 255.0  # TODO too high ?
    return img, ndvi, mask


def loadStaticData(normalize=False):
    altitude = io.imread(fld + alt).astype(np.float32)
    slope = io.imread(fld + slp).astype(np.float32)
    if normalize:
        altitude /= 2555.0
        slope /= 52.0
    return altitude, slope


def loadLandslideDataset():
    satelite_images = {}
    for idx, img_name in enumerate(sateliteImages):
        satelite_images[idx] = {
            "img": io.imread(fld + img_name + ".tif").astype(np.float32),
            "ndvi": io.imread(fld + img_name + "_NDVI.tif").astype(np.float32),
            "mask": io.imread(fld + img_name + "_mask_ls.tif").astype(np.float32)
        }
    altitude, slope = loadStaticData()

    return satelite_images, altitude, slope

def LandslideGenerator():
    # load data

    # generate coordinates (one for each set of lables)

    # sample ratio of p from s1 and (1-p) from s2

    # return combined batch
    pass

def getTrainDataForDir(year=2, areaSize=8, seed=1, numNeg=100):
    altMatrix = np.array(io.imread(fld + alt), dtype=np.float32) / 2555.0
    slpMatrix = np.array(io.imread(fld + slp), dtype=np.float32) / 52.0

    orgi = np.array(io.imread(fld + fimgs[year]), dtype=np.float32) / 20000.0
    addi = np.array(io.imread(fld + nimgs[year]), dtype=np.float32) / 20000.0

    porgi = np.array(io.imread(fld + fimgs[year - 1]), dtype=np.float32) / 20000.0
    paddi = np.array(io.imread(fld + nimgs[year - 1]), dtype=np.float32) / 20000.0

    resi = io.imread(fld + masks[year])

    (posXIDs, posYIDs) = np.where(resi == 1.0)
    (negXIDs, negYIDs) = np.where(resi == 0.0)

    # create train data for one episode
    # create positive instances

    diffSize = areaSize // 2

    numPos = len(posXIDs)
    posIDs = np.arange(numPos)
    negIDs = np.arange(len(negXIDs))
    random.seed(seed)
    random.shuffle(negIDs)
    useNegIDs = np.min([len(negIDs), numNeg * numPos])
    negIDs = negIDs[0:useNegIDs]
    numInstances = len(posIDs) + len(negIDs)
    label = np.zeros([numInstances, 1])
    dataPre = np.zeros([numInstances, areaSize, areaSize, orgi.shape[2]])
    dataPost = np.zeros([numInstances, areaSize, areaSize, orgi.shape[2]])
    dataSLP = np.zeros([numInstances, 1])
    dataALT = np.zeros([numInstances, 1])
    dataNDVI = np.zeros([numInstances, 1])
    counter = 0
    for posID in posIDs:
        label[counter] = 1.0
        curX = posXIDs[posID]
        curY = posYIDs[posID]
        orgiTmp = orgi[curX - diffSize:curX + diffSize, curY - diffSize:curY + diffSize, :]
        dataPost[counter, 0:orgiTmp.shape[0], 0:orgiTmp.shape[1], 0:orgiTmp.shape[2]] = orgiTmp
        preTmp = porgi[curX - diffSize:curX + diffSize, curY - diffSize:curY + diffSize, :]
        dataPre[counter, 0:orgiTmp.shape[0], 0:orgiTmp.shape[1], 0:orgiTmp.shape[2]] = preTmp
        dataALT[counter] = altMatrix[curX, curY]
        dataSLP[counter] = slpMatrix[curX, curY]
        dataNDVI[counter] = addi[curX, curY] - paddi[curX, curY]
        counter += 1

    for negID in negIDs:
        label[counter] = 0.0
        curX = negXIDs[negID]
        curY = negYIDs[negID]
        orgiTmp = orgi[curX - diffSize:curX + diffSize, curY - diffSize:curY + diffSize, :]
        dataPost[counter, 0:orgiTmp.shape[0], 0:orgiTmp.shape[1], 0:orgiTmp.shape[2]] = orgiTmp
        preTmp = porgi[curX - diffSize:curX + diffSize, curY - diffSize:curY + diffSize, :]
        dataPre[counter, 0:orgiTmp.shape[0], 0:orgiTmp.shape[1], 0:orgiTmp.shape[2]] = preTmp
        dataALT[counter] = altMatrix[curX, curY]
        dataSLP[counter] = slpMatrix[curX, curY]
        dataNDVI[counter] = addi[curX, curY] - paddi[curX, curY]
        counter += 1
    return (dataPre, dataPost, dataALT, dataSLP, dataNDVI, label)


if __name__ == "__main__":
    # parameter
    areaSize = 4
    seed = 1
    numNeg = 10
    trainYears = [1, 2, 3, 4, 5]
    testYear = 6
    # create validation set
    # (dataPreTest, dataPostTest, dataALTTest, dataSLPTest, dataNDVITest, labelTest) = getTrainDataForDir(year=testYear,
    #                                                                                                    areaSize=areaSize,
    #                                                                                                    seed=seed,
    #                                                                                                    numNeg=numNeg)
    print("All Done!")
