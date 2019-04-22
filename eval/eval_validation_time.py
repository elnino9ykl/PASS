# Code to calculate the validation time through the pass dataset
# April 2019
# Kailun Yang
#######################

import numpy as np
import torch
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet_pspnet_sppad import Net
from transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 28

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024*4),Image.BILINEAR), #4 times corrspond to using 4 feature models
    ToTensor(),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        
        for a in own_state.keys():
            print(a)
        for a in state_dict.keys():
            print(a)
        print('-----------')
        
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
      
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()
    
    time_all= []
    with torch.no_grad():
        for step, (images, filename) in enumerate(loader):
            
            images = images.cuda()
            start_time = time.time()
            outputs = model(images)
            fwt = time.time() - start_time
            time_all.append(fwt)

            label = outputs[0].cpu().max(0)[1].data.byte()
            label_color = Colorize()(label.unsqueeze(0))
        
            filenameSave = "./save_color/" + filename[0].split("leftImg8bit/")[1]
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
            label_save = ToPILImage()(label_color)
             
            label_save.save(filenameSave) 
            
            print (step, filenameSave)
            print ("FPS (Mean: %.4f)" % (1.0000/ (sum(time_all) / len(time_all))))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfpspnet.pth")
    parser.add_argument('--loadModel', default="erfnet_pspnet_sppad.py") #can be other networks, such as erfnet_apspnet_sppad.py, edanet_sppad.py, cgnet_sppad.py, linknet_sppad.py, sqnet_sppad.py
    parser.add_argument('--subset', default="pass")  #can be val, test, train, pass, demoSequence

    parser.add_argument('--datadir', default="../dataset/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
