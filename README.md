# gbdx-caffe
Running a simple caffe-based object detector on GBDX using the GPU

## Build

The detector is based on a docker image that provides GPU support for caffe on gbdx. Instructions on how to create this runtime image from scratch and/or customize it can be found at

https://github.com/ctuskdg/gbdx-gpu-docker

Creating the gbdx-caffe task is then a simple `docker build -t my-task app`

## Test

A simple testrun can be performed using the supplied test image and model. Copy the image and model directories that can be found in the work directory to a location in your gbdx storage space, then edit the supplied workflow.json example file accordingly to adjust input and output locations.

## Running your own detections

We recommend training your model using nvidia-digits. The model format the task expects follows the format digits uses. Simply train your model , upload it to your gbdx storage area and adjust the task and its parameters to use the new model. Make sure to adjust the pyramid_window_sizes and pyramid_step_sizes parameters according to the pixel sizes of the objects you trained on. 
For example for detection of commercial airliners on  pan-sharpened imagery pyramid_window_sizes [150,100] and pyramid_step_sizes [40,30] are meaningful settings.

