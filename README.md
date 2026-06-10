The hyperparameter settings of NECE-GCN are configured by loading the .py file located in NECE-GCN/configs/gcn/ntu_mutual/mutual. The preprocessing scripts are located in the .py files under NECE-GCN/src/dataset and are automatically invoked when running main.py.

The datasets used in this paper are available at the following links:

NTU-RGB+D & NTU-RGB+D 120:
https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
(Since this study only uses skeleton data, it is sufficient to download the skeleton data from NTU-RGB+D. The download file is nturgbd_skeletons_s001_to_s017.zip. After extraction, it contains the complete NTU-RGB+D skeleton data.)

https://drive.google.com/open?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w
(This link provides the file nturgbd_skeletons_s018_to_s032.zip, which is an extended dataset of NTU-RGB+D. The NTU-RGB+D 120 dataset consists of nturgbd_skeletons_s001_to_s017.zip and nturgbd_skeletons_s018_to_s032.zip. To use the NTU-RGB+D 120 dataset, both nturgbd_skeletons_s001_to_s017.zip and nturgbd_skeletons_s018_to_s032.zip should be combined.)

SBU-Kinect-Interaction dataset:
http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt

