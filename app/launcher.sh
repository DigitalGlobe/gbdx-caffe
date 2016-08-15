#!/bin/bash

# Set up all environment variables here
export PATH="/usr/local/cuda/bin:/opt/caffe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="/opt/caffe/lib/python2.7/site-packages:/opt/caffe/lib64/python2.7/site-packages:/opt/caffe/python"
export LD_LIBRARY_PATH="/opt/caffe/lib64"
# Launch application
mkdir -p /mnt/work/output/log
/opt/caffe/bin/python app.py > /mnt/work/output/log/launcher.log 2>&1 
