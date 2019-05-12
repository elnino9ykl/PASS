# Training ERF-PSPNet or ERF-APSPNet in Pytorch

PyTorch code for training ERF-PSPNet or ERF-APSPNet models. The code was based initially on the code from [bodokaiser/piwise](https://github.com/bodokaiser/piwise), adapted with several custom added modifications and tweaks. Some of them are:
- Load dataset (Cityscapes, Mapillary Vistas or other datasets)
- ERF-PSPNet and ERF-APSPNet models definition
- Show IoU and ACC on each epoch during training (adapted using cityscapes scripts)
- Three different loss functions including cross-entropy, focal-loss and lovasz-loss
- Save snapshots and best model during training
- Save additional output files useful for checking results (see below "Output files...")
- Resume training from checkpoint (use "--resume" flag in the command)

## Options
For all options and defaults please see the bottom of the "main.py" file. Required ones are --savedir (name for creating a new folder with all the outputs of the training) and --datadir (path to cityscapes directory).

## Example commands

Train ERF-PSPNet decoder using encoder's pretrained weights with ImageNet:
```
python3.6 main.py --savedir erfnet_pspnet_training1 --datadir /home/kailun/Downloads/PASS-master/dataset/ --num-epochs 60 --batch-size 6 --decoder --pretrainedEncoder "../trained_models/erfnet_encoder_pretrained.pth.tar"
```

Train ERF-APSPNet decoder using encoder's pretrained weights with ImageNet:
```
python3.6 main.py --savedir erfnet_apspnet_training1 --datadir /home/kailun/Downloads/PASS-master/dataset/ --num-epochs 60 --batch-size 6 --decoder --pretrainedEncoder "../trained_models/erfnet_encoder_pretrained.pth.tar" --model erfnet_apspnet
```

Train encoder with 60 epochs and batch=6 and then train decoder (decoder training starts after encoder training):
```
python3.6 main.py --savedir erfnet_pspnet_traning1 --datadir /home/kailun/Downloads/PASS-master/dataset/ --num-epochs 60 --batch-size 6 
```

## Output files generated for each training:
Each training will create a new folder in the "../save/" directory named with the parameter --savedir and the following files:
* **automated_log.txt**: Plain text file that contains in columns the following info of each epoch {Epoch, Train-loss,Test-loss,Train-IoU,Test-IoU, learningRate}. Can be used to plot using Gnuplot or Excel.
* **best.txt**: Plain text file containing a line with the best IoU achieved during training and its epoch.
* **checkpoint.pth**: bundle file that contains the checkpoint of the last trained epoch.
* **{model}.py**: copy of the model file used (default erfnet_pspnet.py). 
* **model.txt**: Plain text that displays the model's layers
* **model_superbest.pth**: saved weights of the epoch that achieved best val accuracy.
* **opts.txt**: Plain text file containing the options used for this training

NOTE: Encoder trainings have an added "_encoder" tag to each file's name.


By default, only Validation IoU is calculated for faster training (can be changed in options)

## Visualization
If you want to visualize the outputs during training add the "--visualize" flag and open an extra tab with:
```
python3.6 -m visdom.server -port 8097
```
The plots will be available using the browser in http://localhost.com:8097

## Multi-GPU
If you wish to specify which GPUs to use, use the CUDA_VISIBLE_DEVICES command:
```
CUDA_VISIBLE_DEVICES=0 python3.6 main.py ...
CUDA_VISIBLE_DEVICES=0,1 python3.6 main.py ...
```


