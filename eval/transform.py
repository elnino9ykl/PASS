# Code with transformations
# April 2019
# Kailun Yang
#######################

import numpy as np
import torch

from PIL import Image

def colormap_mapillary(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([153,153,153])
    cmap[1,:] = np.array([210,170,100])
    cmap[2,:] = np.array([220,220,220])
    cmap[3,:] = np.array([250,170, 30])
    cmap[4,:] = np.array([  0,  0,142])
    cmap[5,:] = np.array([  0,  0, 70])

    cmap[6,:] = np.array([119, 11, 32])
    cmap[7,:] = np.array([  0,  0,230])
    cmap[8,:] = np.array([  0, 60,100])
    cmap[9,:] = np.array([220,220,  0])
    cmap[10,:]= np.array([192,192,192])

    cmap[11,:]= np.array([128, 64,128])
    cmap[12,:]= np.array([244, 35,232])
    cmap[13,:]= np.array([170,170,170])
    cmap[14,:]= np.array([140,140,200])
    cmap[15,:]= np.array([128, 64,255])

    cmap[16,:]= np.array([196,196,196])
    cmap[17,:]= np.array([190,153,153])
    cmap[18,:]= np.array([102,102,156])
    cmap[19,:]= np.array([ 70, 70, 70])

    cmap[20,:]= np.array([220, 20, 60])
    cmap[21,:]= np.array([255,  0,  0])
    cmap[22,:]= np.array([ 70,130,180])
    cmap[23,:]= np.array([107,142, 35])
 
    cmap[24,:]= np.array([152,251,152])
    cmap[25,:]= np.array([255,255,255])
    cmap[26,:]= np.array([200,128,128])
    cmap[27,:]= np.array([  0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=28):
        #self.cmap = colormap(256)
        self.cmap = colormap_mapillary(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
