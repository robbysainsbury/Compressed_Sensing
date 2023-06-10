import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import copy 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import hydroeval as he

np.random.seed(0)

class CompresedSensingInterpolator():
    def __init__(self):
        self.dataMean = None
        self.timeSeriesLength = None        

    def _normalize(self, data):
        mean = np.nanmean(data)
        self.dataMean = mean
        data = data - self.dataMean
        return data

    def _denormalize(self, data):
        if self.dataMean == None: # if _normalize hasn't been called yet
            return data
        data = data + self.dataMean
        return data

    def _getDesignMatrix(self, ts, numBases):
        A = []
        # sample up to the nyquist frequency
        omegas = np.linspace(1, self.timeSeriesLength // 2, numBases)
        loop = tqdm(total = omegas.shape[0], desc = "creating design matrix")
        for omega in omegas:
            bi_r = np.cos(omega * ts)
            bi_i = np.sin(omega * ts)
    
            A.append(bi_r)
            A.append(bi_i)
            loop.update()
        loop.close()
    
        A = np.array(A).T

        return A
    
    def _getDataWithoutNones(self,dataWithNones, ts):
        '''
        use all the data that is not None 
        '''
        mask = np.isnan(dataWithNones)
        dataWithNones = np.array(dataWithNones)
        data = dataWithNones[~mask]

        tsWithoutNones = ts[~mask]
        return data, tsWithoutNones
    
    def _getTs(self, dataWithNones):
        ''' 
        frame the whole time series as occuring across 2 pi 
        (this makes the math for basis functions easy)
        ''' 
        self.timeSeriesLength = dataWithNones.shape[0]
        ts = np.linspace(0, 2 * np.pi, self.timeSeriesLength)
        return ts
    
    def _predict(self, A, coefficients):
        return A @ coefficients
    
    def _getFirstGuessOfCoefficients(self, data, A):
        result = np.linalg.lstsq(A, data, rcond=None)
        coefficients = result[0]
        return coefficients
    
#    def _getL1Norm(self, coefficients):
#        return torch.linalg.norm(coefficients, 1)

    def _getCoefficients(self, dataWithoutNones, tsWithoutNones, numBases, method="SLSQP"):
        A = self._getDesignMatrix(tsWithoutNones, numBases)
        firstGuess = self._getFirstGuessOfCoefficients(dataWithoutNones, A)

        # convert everything over to tensors to user Pytorch
        A = torch.Tensor(A)
        y = torch.Tensor(dataWithoutNones)
        thetas = torch.nn.Parameter(torch.Tensor(firstGuess))

        optimTheta = optim.Adam([thetas], lr=1e-1)

        numSteps = 1000
        lamda = 0.5
        loop = tqdm(total=numSteps, desc="optimizing parameters")
        for i in range(numSteps):
            # clear the gradient
            optimTheta.zero_grad()

            # make predictions
            yHat = A @ thetas

            # measure loss
            loss1 = torch.linalg.norm(y - yHat, 2) # 2-norm of the error
            loss2 = torch.linalg.norm(thetas, 1) # 1-norm of the parameters
            loss = loss1 + (lamda * loss2)

            # calculate backward pass
            loss.backward()
            optimTheta.step()
            loop.update()
        loop.close()

        coefficients = thetas.detach().numpy()

        return coefficients


    def _getReconstruction(self, coefficients, dataWithNones, ts, numBases, ys):
        # only model the missing times
        mask = np.isnan(dataWithNones)
        tsWithNones = ts[mask]
        A = self._getDesignMatrix(tsWithNones, numBases)
        reconstructedMissingData = self._predict(A, coefficients)
        reconstructedFullData = copy.copy(dataWithNones)
        reconstructedFullData[mask] = reconstructedMissingData
        return reconstructedFullData
        
    def interpolate(self, dataWithNones, numBases=100, ys=None, method="SLSQP"):
        '''
        uses comprssed sensing to interpolate missing data
    
        numBases: the number of fourier basis functions to use
        when building the interpolation model. More = more accurate.
        '''
        normalizedDataWithNones = self._normalize(dataWithNones)

        ts = self._getTs(normalizedDataWithNones)
        dataWithoutNones, tsWithoutNones = self._getDataWithoutNones(normalizedDataWithNones, ts)
        coefficients = self._getCoefficients(dataWithoutNones, tsWithoutNones, numBases, method=method)
        normalizedReconstruction = self._getReconstruction(coefficients, normalizedDataWithNones, ts, numBases, ys)

        reconstruction = self._denormalize(normalizedReconstruction)

        return reconstruction

