#!/bin/sh
apt-get update
apt-get --yes install python3
apt-get --yes install python3-pip
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
#dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
#apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#apt-get update
#http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
#apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
#apt-get update
#apt-get install --yes --no-install-recommends nvidia-driver-418
nvidia-smi
#apt-get install --yes --no-install-recommends \
#    cuda-10-0 \
#    libcudnn7=7.6.2.24-1+cuda10.0  
#apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
#    libnvinfer-dev=5.1.5-1+cuda10.0
pip3 install numpy 
pip3 install pandas
#pip3 uninstall tensorflow
#pip3 install --upgrade tensorflow-gpu
#pip3 install keras
pip3 install sklearn
pip3 install matplotlib
pip3 install cachetools
pip3 install pydicom
apt-get --yes install git
cd $2
git pull 
python3 $1
