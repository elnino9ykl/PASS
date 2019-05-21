# Functions for evaluating/visualizing the network's output

## eval_color.py

This code can be used to produce segmentation in color for visualization purposes. By default it saves images in eval/save_color/folder.

**Options:** Specify the folder path with '--datadir' option. Select the subset with '--subset' ('val', 'train', 'test', 'pass', 'demoSequence').

Specify the trained moddel with '--loadWeights' and '--loadModel' options.

**Examples**
```
python3.6 eval_color.py --datadir /home/kailun/Downloads/PASS-master/dataset/ --subset val --loadDir ../trained_models/ --loadWeights erfpspnet.pth --loadModel erfnet_pspnet.py
```
```
python3.6 eval_color.py --datadir /home/kailun/Downloads/PASS-master/dataset/ -subset pass --loadDir ../trained_models/ --loadWeights erfapspnet.pth --loadModel erfnet_apspnet.py
```

## eval_color_4.py

This code can be used to produce segmentation in color using adapted ERF-PSPNet, ERF-APSPNet or other adapted networks. (using 4 feature models and 1 fusion model, with special padding and upsamling operations)

**Examples**
```
python3.6 eval_color_4.py --datadir /home/kailun/Downloads/PASS-master/dataset/ --subset pass --loadDir ../trained_models/ --loadWeights erfpspnet.pth --loadModel erfnet_pspnet_splits4.py
```

## eval_validation_time.py

This code can be used to calculate the evaluation time running through the PASS dataset.

**Examples**
```
python3.6 eval_validation_time.py --datadir /home/kailun/Downloads/PASS-master/dataset/ --subset pass --loadDir ../trained_models/ --loadWeights erfpspnet.pth --loadModel erfnet_pspnet_sppad.py
```
