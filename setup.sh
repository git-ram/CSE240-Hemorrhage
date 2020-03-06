#!/bin/sh
apt-get update
apt-get --yes install python3
apt-get --yes install python3-pip
pip3 install numpy 
pip3 install pandas
pip3 install keras
pip3 install pydicom
pip3 install tensorflow
pip3 install sklearn
pip3 install matplotlib
pip3 install cachetools
apt-get --yes install git
python3 $1
