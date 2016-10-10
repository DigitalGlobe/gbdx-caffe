#!/usr/bin/env python
import argparse
import os
import sys
import time
import glob
import datetime
import numpy as np
import json
import caffe
from caffe.proto import caffe_pb2
import multiprocessing as mp
import gdal
from gdalconst import *
import osr
import logging as log
import re
from string import Template
from functools import partial
import zipfile
import traceback
import ast

# For profiling the code
#from profilehooks import profile
# log.getLogger().setLevel(log.DEBUG)

# Suppress most caffe output
os.environ['GLOG_minloglevel'] = '2' 

POLYGON_TEMPLATE = Template("POLYGON (($left $bottom, $left $top, $right $top, $right $bottom, $left $bottom))")
LABEL_ID_REGEX = re.compile('^n\d+\s*')

BASE_DIR='/mnt/work/'
INPUT_DIR_PATH = BASE_DIR+'input'
DEFAULT_STATUS_JSON_PATH = BASE_DIR+'status.json'
OUTPUT_VECTORS_DIR_PATH = BASE_DIR+'output/result/detections'
OUTPUT_VECTORS_FILE = 'detection-results.json'
OUTPUT_VECTORS_ZIP_PATH = BASE_DIR+'output/result/detections.zip'

RASTER_DIM = 252 + 28
RASTER_STRIDE = RASTER_DIM - 28

DEFAULT_THRESHOLD = 80.0
DEFAULT_WIN_SIZE = 150
DEFAULT_STEP_SIZE = 30
DEFAULT_MIN_PYRAMID_SIZE = 30
DEFAULT_PYRAMID_SCALE_FACTOR = 1.5
DEFAULT_GPU_FLAG = 'False'
DEFAULT_NUM_PROCESSES = 1

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

status_dict = {
    'status': STATUS_SUCCESS,
    'reason': "Detection succeeded"
}

TIF_SUFFIX = ".tif"

CAFFEMODEL_SUFFIX = ".caffemodel"
DEPLOY_FILE_SUFFIX = "deploy.prototxt"
MEAN_FILE_SUFFIX = "mean.binaryproto"
LABELS_FILE_SUFFIX = "labels.txt"

CAFFE_GPU_BATCH_SIZE= 80
CAFFE_CPU_BATCH_SIZE= 40

def caffe_ms_band_window_transform(window, transformed_size, transformed_window, mean):
    """
    @param window a (width, height, num_channels) numpy array dtype=int or float
    @param transformed_size A tuple (trans_width, trans_height) representing the size of the transfomred image
    @param transformed window (num_channels, trans_width, trans_height) numpy array dtype must be float (32 or 64)
    """

    for i_band in range(window.shape[0]):
        transformed_window[i_band,:,:] = scipy.misc.imresize(window[i_band,:,:], transformed_size, interp='bilinear')
        transformed_window[i_band,:,:] -= mean[i_band]

def caffe_window_transform_bw(window, transformed_size, transformed_window_bw, mean):
    """
    @param window a (width, height, num_channels) numpy array dtype=int or float
    @param transformed_size A tuple (trans_width, trans_height) representing the size of the transfomred image
    @param transformed window (num_channels, trans_width, trans_height) numpy array dtype must be float (32 or 64)
    """
    # Convert RGB to black and white
    # ITU-R 601-2 luma transform:
    #   R*299./1000 + G*587./1000 + B*114./1000 (to match PIL.Image)
    luma_coefs = [.299, .587, .114]
    num_channels = window.shape[2]
    transformed_window = np.zeros(( (num_channels,) + transformed_size), dtype=np.float32)
    if num_channels == 3:
        transformed_window_bw[0,:,:] = 0.
        for i_band in range(window.shape[0]):
            transformed_window_bw[0,:,:] += luma_coefs[i_band]*scipy.misc.imresize(window[i_band,:,:], transformed_size, interp='bilinear')

    else:
        transformed_window_bw[0,:,:] = scipy.misc.imresize(window[i_band,:,:], transformed_size, interp='bilinear')

    # Subtract the mean
    transformed_window_bw -= mean[0]

class GDALImage:
    def __init__(self, imagefile, tilewidth=256, tileheight=256, strideX=None, strideY=None, bands=None, padWithZeros=False):
        self.imagefile = imagefile
        self.tilewidth = tilewidth
        self.tileheight = tileheight
        # Open dataset
        self.dataset = gdal.Open(self.imagefile, gdal.GA_ReadOnly)
        self.nbands = self.dataset.RasterCount
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.geoTransform = self.dataset.GetGeoTransform()
        self.projRef = self.dataset.GetProjectionRef()
        self.datatype = self.dataset.GetRasterBand(1).DataType
        self.isByteImage = ( self.datatype == GDT_Byte )
        self.padWithZeros = padWithZeros

        self.strideX = strideX        
        if strideX == None:
            self.strideX = self.tilewidth

        self.strideY = strideY        
        if strideY == None:
            self.strideY = self.tileheight
                        
        # Set up projections
        self.spr = osr.SpatialReference( self.projRef )
        self.geospr = self.spr.CloneGeogCS()
                
        self.coordTfProjToGeo =  osr.CoordinateTransformation( self.spr, self.geospr )
        self.coordTfGeoToProj =  osr.CoordinateTransformation( self.geospr, self.spr )
        
        # Set up boundingBox
        self.bb_x0 = 0
        self.bb_y0 = 0
        self.bb_x1 = self.width
        self.bb_y1 = self.height

        self.bands = []
        if not bands is None:
            # Verify that the bands are all less than the number of bands
            for i_band in bands:
                if i_band < self.nbands:
                    self.bands.append(i_band)
                else:
                    error_msg = "Error: band {} not in image {}".format(str(i_band), imagefile)
                    log.error(error_msg)
                    raise RuntimeError(error_msg, e)
        else: 
            self.bands = [i for i in range(self.nbands)]
        # Convert to immutable tuple
        self.bands = tuple(self.bands)

    def setBoundingBox(self,x0,y0,x1,y1):
        self.bb_x0 = int ( max( 0, x0 ) )
        self.bb_y0 = int ( max( 0, y0 ) )
        self.bb_x1 = int ( min( x1, self.width ) )
        self.bb_y1 = int ( min( y1, self.height ) )
        
    def setGeoBoundingBox(self,lon_ul,lat_ul,lon_lr,lat_lr):
        x0,y0 = self.tfGeoToRaster(lon_ul, lat_ul)
        x1,y1 = self.tfGeoToRaster(lon_lr, lat_lr)
        self.setBoundingBox( min(x0,x1), min(y0,y1), max(x0,x1), max (y0,y1) )

    def __str__(self):
        return "%d,%d,%d" % ( self.width, self.height, self.nbands)

    def nextTile(self):
        y0 = self.bb_y0
        while y0 < self.bb_y1:
            x0 = self.bb_x0
            y1 = min ( y0+self.tileheight, self.bb_y1 )            
            while x0 < self.bb_x1:
                x1 = min ( x0+self.tilewidth, self.bb_x1 )                
                yield x0, y0, x1, y1
                x0 = x0 + self.strideX
            y0 = y0 + self.strideY

    def nextDataTile(self):
        for x0, y0, x1, y1 in self.nextTile():
            yield self.readTile(x0, y0, x1, y1), x0, y0

    def readTile(self, x0, y0, x1, y1):
        data = self.dataset.ReadAsArray(x0, y0, x1-x0, y1-y0)

        if len(data.shape) == 2: # only one band - extend to 3-dim
            data = np.reshape(data, (1, data.shape[0], data.shape[1]))
        else:
            data = data[self.bands,:,:]

        if self.padWithZeros:
            if ( data.shape[1] < self.tileheight or data.shape[2] < self.tilewidth ):
                tile = np.zeros( ( data.shape[0], self.tileheight, self.tilewidth), dtype=data.dtype )
                tile[:,0:data.shape[1],0:data.shape[2]] = data[:]
                data = tile
        return data
                    
    def tfRasterToProj(self, x,y):
        dfGeoX = self.geoTransform[0] + self.geoTransform[1] * x + self.geoTransform[2] * y;
        dfGeoY = self.geoTransform[3] + self.geoTransform[4] * x + self.geoTransform[5] * y;
        return dfGeoX, dfGeoY
    
    def tfProjToRaster(self, projX, projY):
        x = ( self.geoTransform[5] * ( projX - self.geoTransform[0] ) - self.geoTransform[2] * ( projY - self.geoTransform[3] ) ) / ( self.geoTransform[5] * self.geoTransform[1] + self.geoTransform[4] * self.geoTransform[2] )
        y = (projY -  self.geoTransform[3] - x*self.geoTransform[4] ) / self.geoTransform[5]
        return x,y
    
    def tfProjToGeo(self, projx, projy):
        return self.coordTfProjToGeo.TransformPoint(projx, projy)
    
    def tfGeoToProj(self, longitude, latitude):
        return self.coordTfGeoToProj.TransformPoint(longitude, latitude)
    
    def tfGeoToRaster(self, longitude, latitude):
        proj = self.tfGeoToProj(longitude, latitude)
        return self.tfProjToRaster(proj[0], proj[1])
        
    def tfRasterToGeo(self,x,y):
        proj = self.tfRasterToProj(x, y)
        return self.tfProjToGeo( proj[0], proj[1] )


class PyramidWindowBatcher(object):

    def __init__(self, pyramid, num_channels, window_shape, num_windows, max_batch_size=4096, mult_size=256, transform=caffe_ms_band_window_transform):
        assert isinstance(pyramid, (Pyramid,))
        self.pyramid = pyramid

        self.mult_size = mult_size

        # floor
        num_mini_batches_max = int(max_batch_size/mult_size)
        num_mini_batches_all = int(num_windows/mult_size)
        # If num_windows isn't a multiple of mult_size
        if num_windows % mult_size > 0:
            num_mini_batches_all += 1

        self.batch_size = min(num_mini_batches_max, num_mini_batches_all)*mult_size

        self.num_channels = num_channels
        self.window_shape = window_shape

        self.window_batch = np.zeros((self.batch_size, self.num_channels) + self.window_shape, dtype=np.float32)
        self.x_vals = np.zeros((self.batch_size), dtype=np.int)
        self.y_vals = np.zeros((self.batch_size), dtype=np.int)
        self.window_sizes = np.zeros((self.batch_size), dtype=np.int)

        # for iterator
        self.current_batch_num = 0

        # for transformer
        self.transform = transform

    def window_iteration(self, image):
        window_counter = 0 
        for win_size, win_step in self.pyramid:
            for (x, y, window) in self.sliding_window(image, window_size=(win_size, win_size), step_size=win_step):

                # Apply the transform to resize and swap axis of the windowed image
                self.transform(window, self.window_shape, self.window_batch[window_counter,:,:,:])
                self.x_vals[window_counter] = x
                self.y_vals[window_counter] = y
                self.window_sizes[window_counter] = win_size
                window_counter += 1
                if window_counter == self.get_batch_size():
                    window_counter = 0
                    yield self.window_batch, self.x_vals, self.y_vals, self.window_sizes, self.get_batch_size()

        if window_counter > 0:
            # batching finished return current state of window_batch
            yield self.window_batch, self.x_vals, self.y_vals, self.window_sizes, window_counter

    def get_batch_size(self):
        return self.batch_size
                
    def sliding_window(self, image, window_size, step_size):
        for y in xrange(0, image.shape[0] - window_size[0] + 1, step_size):
            for x in xrange(0, image.shape[1] - window_size[1] + 1, step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def iter_batches(self, image):
        return self.window_iteration(image)

class CaffeBatchClassifier(object):
    def __init__(self,  caffe_models, deploy_files, label_files, mean_files, gpu_flag='False'):
        self.caffe_net = None
        self.caffe_net_models = []
        self.caffe_models = caffe_models
        self.deploy_files = deploy_files
        self.label_files = label_files
        self.mean_files = mean_files

        # Perform checks on these arguments
        self.check_valid()

        self.num_models = len(self.caffe_models)
        if gpu_flag.lower() == 'true':
            self.caffe_batch_size = CAFFE_GPU_BATCH_SIZE
        else:
            self.caffe_batch_size = CAFFE_CPU_BATCH_SIZE

        # Set gpu flags
        if gpu_flag.lower() == 'true':
            self.gpu_flag = True
            caffe.set_mode_gpu()
        else:
            self.gpu_flag = False
            caffe.set_mode_cpu()

        self.loaded_model = -1
        self.load_all_models()
        self.setup_transformer()
        #keep top 5 classes if training model contains more than 5 classes. Otherwise keep all classes.
        self.top_n = 5 if len(self.labels) >5 else len(self.labels)
        
    def check_valid(self):
        pass

    def get_num_models(self):
        return self.num_models

    def get_caffe_model(self, model_num):
        return self.caffe_models[model_num]

    def get_deploy_file(self, model_num):
        return self.deploy_files[model_num]

    def get_label_file(self, model_num):
        return self.label_files[model_num]

    def get_mean_file(self, model_num):
        return self.mean_files[model_num]

    def get_loaded_model_num(self):
        return self.loaded_model

    def set_loaded_model_num(self, model_num):
        self.loaded_model = model_num

    def get_transformer(self,):
        return self.transformer

    def setup_transformer(self,):
        # Function pointer
        with open(self.get_mean_file(0)) as infile:
            blob = caffe_pb2.BlobProto()
            data = infile.read()
            blob.ParseFromString(data)
            arr = np.array(caffe.io.blobproto_to_array(blob))
            mean = arr[0].mean(1).mean(1)

        if self.get_caffe_num_channels() == 1:
            self.transformer = partial(caffe_window_transform_bw, mean=mean)
        else:
            # Use specified bands
            self.transformer = partial(caffe_ms_band_window_transform, mean=mean)

    def load_all_models(self):

        for model_num in range(self.get_num_models()):
            caffe_net = caffe.Net(self.get_deploy_file(model_num), 
                                  self.get_caffe_model(model_num), 
                                  caffe.TEST)
            if model_num == 0:
                # Size of output
                self.caffe_input_shape = caffe_net.blobs['data'].data.shape
                self.set_caffe_num_channels(self.caffe_input_shape[1])
                self.set_caffe_window_size((self.caffe_input_shape[2], 
                                            self.caffe_input_shape[3]))
                caffe_output_shape = caffe_net.blobs['prob'].data.shape
                self.set_caffe_output_size(caffe_output_shape[-1])
                self.labels = read_labels(self.get_label_file(model_num))
                self.caffe_start_ind = 0
                self.caffe_end_ind = len(caffe_net.layers) - 1
                                
            caffe_net.blobs['data'].reshape(self.get_caffe_batch_size(),
                                            self.get_caffe_num_channels(),
                                            self.get_caffe_window_size()[0],
                                            self.get_caffe_window_size()[1])
            caffe_net.reshape()

            self.caffe_net_models.append(caffe_net)

    def get_caffe_batch_size(self):
        return self.caffe_batch_size

    def get_caffe_num_channels(self):
        return self.num_channels

    def get_caffe_window_size(self):
        return self.window_size

    def set_caffe_batch_size(self, batch_size):
        self.caffe_batch_size = batch_size

    def set_caffe_num_channels(self, num_channels):
        self.num_channels = num_channels

    def set_caffe_window_size(self, size_tuple):
        self.window_size = size_tuple

    def set_caffe_output_size(self, output_size):
        self.caffe_output_size = output_size

    def get_caffe_output_size(self):
        return self.caffe_output_size

    def get_scores_model(self, window_batch, model_num):

        if model_num == 0:
            self.caffe_net_models[model_num].blobs['data'].data[...] = window_batch
        else:
            self.caffe_net_models[model_num].blobs['data'].data[...] = self.caffe_net_models[0].blobs['data'].data[...]
            
        out = self.caffe_net_models[model_num].forward()
        return out['prob']

    def classify_batch_all_models(self, window_batch, num_windows):

        # Classify the image
        batch_scores = np.zeros((self.num_models, num_windows, self.get_caffe_output_size()))

        num_mini_batches = int(np.ceil(np.float(num_windows)/self.caffe_batch_size))
        for i in range(num_mini_batches):
            batch_start = i*self.caffe_batch_size
            batch_end = (i+1)*self.caffe_batch_size
            if batch_end > num_windows:
                batch_end_resize = num_windows
            else:
                batch_end_resize = batch_end
            classify_start_time = time.time()

            # Perform classification on the full mini batch size
            for model_num in range(self.num_models):
                result = self.get_scores_model(window_batch[batch_start:batch_end,:,:,:], model_num)

                # Store only the valid scores
                batch_scores[model_num, batch_start:batch_end_resize,:] = result[0:batch_end_resize-batch_start,:]

            log.info('Classification of batch size {} took {} seconds.'.format(self.caffe_batch_size, (time.time() - classify_start_time)))

        return batch_scores

    def classify_batch(self, batch, batch_size):
        class_names = np.empty((batch_size, self.num_models, self.top_n), dtype="S32")
        class_vals =  np.zeros((batch_size, self.num_models, self.top_n), dtype=np.float32)
        scores = self.classify_batch_all_models(batch, batch_size)
        for model_num in range(self.num_models):
            # TODO: Can we improve data access?
            class_names[:, model_num, :], class_vals[:, model_num, :] = classify_from_scores(scores[model_num,:,:], self.labels)

        return class_names, class_vals


def read_labels(labels_file):
    if not labels_file:
        log.info('WARNING: No labels file provided. Results will be difficult to interpret.')
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    if len(labels) == 0:
        raise ValueError("No labels found")
    return labels

def classify_from_scores(scores, labels):
    top_n = 5
    if scores.shape[1] < top_n:
        top_n = scores.shape[1]
    indices = (-scores).argsort()[:, 0:top_n] # take top n results
    num_windows = scores.shape[0]
    class_names = np.empty((num_windows, top_n), dtype="S32")
    class_vals =  np.zeros((num_windows, top_n), dtype=np.float32)
    for i_window in range(num_windows):
        for i_index in range(top_n):
            index = indices[i_window, i_index]
            label = labels[index]
            class_names[i_window, i_index] = label
            class_vals[i_window, i_index] = round(100.0*scores[i_window, index], 4)

    return class_names, class_vals

def process_results(gdal_image, x0, y0, x_vals, y_vals, 
                    window_sizes, image_name, item_date, sat_id, cat_id,
                    num_ensembles, caffemodels,
                    threshold, class_names, class_vals, threshold_dict, output_vectors_dir):

    # Get the model names
    model_names = []
    for caffemodel in caffemodels:
        model_names.append(get_model_name_from_path(caffemodel))
    num_windows = class_names.shape[0]
    tn = 5 if class_names.shape[2] >5 else class_names.shape[2] 
    for i_window in range(num_windows):
        all_match = True
        for i_ens in range(1, num_ensembles):
            all_match = all_match and class_names[i_window, 0, 0] == class_names[i_window, i_ens, 0]

        if not all_match:
            log.info("Ensemble differs")
            continue

        top_cat = class_names[i_window, 0, 0]
        avg = 0.0
        for i_ens in range(num_ensembles):
            avg += class_vals[i_window, i_ens, 0]
        avg /= num_ensembles

        log.info("Found match for category '%s' with avg score: %s" % (top_cat, avg))

        if avg >= threshold_dict[top_cat.lower()]:
            x = x_vals[i_window]
            y = y_vals[i_window]
            window_size = window_sizes[i_window]

            top_n_geojson = []
            for i_ens in range(num_ensembles):
                ens_top_n = []
                for i_n in range(tn):
                    top = [class_names[i_window, i_ens, i_n], float(class_vals[i_window, i_ens, i_n])]
                    ens_top_n.append(top)
                top_n_geojson.append(ens_top_n)

            geom = get_polygon_array(gdal_image, x0, y0, x, x + window_size, y, y + window_size)
            geojson_item = generate_geojson_item(geom, top_n_geojson, model_names, image_name, item_date, sat_id, cat_id, caffemodels)
            geojson_item = dict(geojson_item)
            write_vector_file(geojson_item, os.path.join( output_vectors_dir, OUTPUT_VECTORS_FILE ) )
        else:
            log.info("Average of the scores is below threshold")

class Pyramid(object):

    def __init__(self, max_window_size=DEFAULT_WIN_SIZE, 
                 max_window_step=DEFAULT_STEP_SIZE,
                 min_window_size=DEFAULT_MIN_PYRAMID_SIZE, 
                 window_scale_factor=DEFAULT_PYRAMID_SCALE_FACTOR, 
                 window_sizes=None,
                 step_sizes=None):
        """
        Constructor
        @param max_window_size size of the largest window
        @param max_window_step largest window step size
        @param max_window_size size of the largest window
        @param max_window_size size of the largest window
        """
        # Specifying windows_sizes overrides other parameters
        if not window_sizes is None:
            assert isinstance(window_sizes, (list, tuple))
            self.num_sizes = len(window_sizes)
            self.window_sizes = np.zeros((self.num_sizes), dtype=np.int)
            self.window_sizes[:] = window_sizes[:]
        else:
            val = np.log(float(min_window_size)/max_window_size)/np.log(1./window_scale_factor)
            self.num_sizes = int(val)+1
            self.window_sizes = np.zeros((self.num_sizes), dtype=np.int)
            self.window_sizes[0] = max_window_size
            for i in range(1, self.num_sizes):
                self.window_sizes[i] = self.window_sizes[i-1]/window_scale_factor

        if not step_sizes is None:
            assert isinstance(step_sizes, (list, tuple))
            self.step_sizes = np.zeros((self.num_sizes), dtype=np.int)
            self.step_sizes[:] = step_sizes[:]
        else:
            self.step_sizes = np.zeros((self.num_sizes), dtype=np.int)
            self.step_sizes[0] = max_window_step
            for i in range(1, self.num_sizes):
                #self.step_sizes[i] = self.step_sizes[i-1]/window_scale_factor
                step = int(self.step_sizes[i-1]/window_scale_factor)
                if step == 0:
                    # Smallest possible step size is one
                    step = 1
                self.step_sizes[i] = step

        self.current = 0

    def calc_pyramiding(self, image_shape):
        window_counter = 0
        pyramid_histogram = {}
        # Iterator over self
        for win_size, win_step in self:
            num_windows = num_sliding_windows(image_shape[1:], step_size=win_step, window_size=(win_size, win_size))
            window_counter += num_windows
            pyramid_histogram[(win_size,win_step)] = num_windows

        num_windows = window_counter
        return num_windows, pyramid_histogram

    def get_window_histogram(self, image_shape):
        _, pyr_hist = self.calc_pyramiding(image_shape)
        return pyr_hist

    def get_num_windows(self, image_shape):
        num_windows, _ = self.calc_pyramiding(image_shape)
        return num_windows

    def __iter__(self):
        return self

    def next(self):
        if self.current < self.num_sizes:
            self.current += 1
            return self.window_sizes[self.current-1], self.step_sizes[self.current-1]
        else:
            self.current = 0
            raise StopIteration

    def get_window_sizes(self):
        return self.window_sizes 

    def get_step_sizes(self):
        return self.step_sizes

def num_sliding_windows(image_shape, step_size, window_size):
    num_windows = ((image_shape[0]-window_size[0])/step_size+1)*((image_shape[1]-window_size[1])/step_size+1)
    return num_windows

def get_polygon_array(gdal_image, x0, y0, pixel_left, pixel_right, pixel_top, pixel_bottom):
    new_left, new_top, _ = gdal_image.tfRasterToGeo(x0+pixel_left, y0+pixel_top)
    new_right, new_bottom, _ = gdal_image.tfRasterToGeo(x0+pixel_right, y0+pixel_bottom)

    return [[[new_left, new_bottom], [new_left, new_top], [new_right, new_top], [new_right, new_bottom],
             [new_left, new_bottom]]]


def convert_model_top_n_to_hash(top_n, model_names):
    top_n_hash = {}
    for i in range(len(top_n)):
        model_name = model_names[i]
        for entries in top_n[i]:
            key = "_".join([model_name, strip_label_id(entries[0])]).replace(" ", "_") + '_dbl'
            top_n_hash[key] = entries[1]
    return top_n_hash


def strip_label_id(value):
    return re.sub(LABEL_ID_REGEX, '', value)


def generate_global_top_n(local_top_ns):
    return generate_global_top_items(local_top_ns)[:5]


def generate_global_top_items(local_top_ns, prefix=None):
    global_top_hash = {}
    for i in range(len(local_top_ns)):
        for j in range(len(local_top_ns[i])):
            label = strip_label_id(local_top_ns[i][j][0])
            if prefix:
                label = prefix + label
                label = label.replace(" ", "_") + '_dbl'
            if label in global_top_hash.keys():
                value = global_top_hash[label]
                if local_top_ns[i][j][1] > value:
                    global_top_hash[label] = local_top_ns[i][j][1]
            else:
                global_top_hash[label] = local_top_ns[i][j][1]
    return sorted(global_top_hash.items(), key=lambda x: x[1], reverse=True)

def get_model_name_from_path(model_path):
    basename = os.path.basename(model_path)
    model_name, extension = os.path.splitext(basename)
    return model_name

def generate_geojson_item(geometry, top_n, model_names, image_name, item_date, sat_id, cat_id, models):
    global_top_n = generate_global_top_n(top_n)
    item_type = global_top_n[0][0]
    item_score = global_top_n[0][1]
    models = [model.split(INPUT_DIR_PATH)[-1] for model in (models or [])] # remove parent directory
    name = os.path.basename(image_name)
    values = []
    for entry in global_top_n:
        values.append(entry[0])
    comment = " ".join(values)
    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': geometry
        },
        'properties': {
            'date': item_date,
            'name': name,
            'type': item_type,
            'score': item_score,
            'comment': comment,
            'source': 'gbdx-caffe',
            'sat_id': sat_id,
            'cat_id': cat_id
        }
    }


def init_vector_dir(output_dir):
    try:
        # create dir if necessary
        if not os.path.exists(output_dir):
            log.info("Creating output directory: " + output_dir)
            os.makedirs(output_dir)
    except Exception, e:
        error_msg = "Encountered exception: " + str(e)
        log.error(error_msg)

def write_vector_file(geojson_item, output_path):
    try:
        out = None
        if not os.path.exists(output_path):
            out = open( output_path, "w")
            out.write('{ "type": "FeatureCollection","features": [''')
        else:
            out = open(output_path, 'a')
            out.write(",")
            
        json.dump(geojson_item, out)
        out.close()
    except Exception, e:
        error_msg = "Encountered exception: " + str(e)
        log.error(error_msg)

def zip_vectors_file(output_dir, zip_path):
    try:
        zip_file = zipfile.ZipFile(zip_path, 'w', allowZip64=True)
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            zip_file.write(file_path, file_name)
            os.remove(file_path)
        zip_file.close()
        os.rmdir(output_dir)
    except Exception, e:
        error_msg = "Encountered exception: " + str(e)
        log.error(error_msg)
        raise RuntimeError(error_msg, e)


def single_tiles_slide_window_classify(gdal_image, image, x0,y0, args,
                                       image_name, item_date, sat_id, cat_id,
                                       mean_files, caffemodels, deploy_files, labels_files,
                                       classifier, gpu_flag, threshold_dict, output_vectors_dir):
    image_shape = image.shape

    # Calculate the window and step sizes for the pyramiding
    pyr = Pyramid(max_window_size=args.win_size, 
                  max_window_step=args.step_size,
                  min_window_size=args.pyramid_min_size,
                  window_scale_factor=args.pyramid_scale_factor,
                  window_sizes=args.pyramid_window_sizes,
                  step_sizes=args.pyramid_step_sizes)

    log.info("Pyramid window sizes: "+ str( pyr.get_window_sizes()) )
    log.info("Pyramid step sizes: " + str( pyr.get_step_sizes()) )

    log.info("Pyramid num_windows: " + str( pyr.get_num_windows(image_shape) ) )
    log.info("Pyramid histogram: " + str( pyr.get_window_histogram(image_shape) ) )
    
    pyr_window_batcher = PyramidWindowBatcher(
               pyr, 
               classifier.get_caffe_num_channels(), 
               classifier.get_caffe_window_size(),
               num_windows=pyr.get_num_windows(image_shape),
               max_batch_size=4096,
               mult_size=classifier.get_caffe_batch_size(),
               transform=classifier.get_transformer())

    for batch, x_vals, y_vals, window_sizes, batch_size in pyr_window_batcher.iter_batches(image):
        # Perform ensemble classification on the batch
        class_names, class_vals = classifier.classify_batch(batch, batch_size)

        # Generate geojson files etc.
        process_results(gdal_image, x0, y0, 
                        x_vals[:batch_size], y_vals[:batch_size], window_sizes[:batch_size], 
                        image_name, item_date, sat_id, cat_id,
                        classifier.get_num_models(), caffemodels, args.threshold, 
                        class_names, class_vals, threshold_dict, output_vectors_dir)


def classify_broad_area_multi_process(gdal_image, image, x0,y0, args,
                                      image_name, item_date, sat_id, cat_id,
                                      mean_files, caffemodels, deploy_files, labels_files,
                                      classifier, gpu_flag, threshold_dict):

    a = datetime.datetime.now()

    single_tiles_slide_window_classify(gdal_image, image, x0,y0, args,
                                       image_name, item_date, sat_id, cat_id,
                                       mean_files, caffemodels, deploy_files, labels_files,
                                       classifier, gpu_flag, threshold_dict, args.output_vectors_dir)
    b = datetime.datetime.now()
    c = b - a
    log.debug("Total Time to process: ["+str(c)+"]")

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", "-t", help="The geotif to perform analysis on", required=True)
    parser.add_argument("--imd", "-i", help="The imd metadata file")
    parser.add_argument("--model_paths", "-m", help="The directory holding the model files", required=True, nargs="+")
    parser.add_argument("--threshold", "-th",
                        help="The probability threshold above which an item will be written to the output",
                        type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--win_size", "-w", help="The window size in pixels", type=int, default=DEFAULT_WIN_SIZE)
    parser.add_argument("--step_size", "-s", help="The step size in pixels", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--status_path", "-sp", help="The output path for the status file", default=DEFAULT_STATUS_JSON_PATH)
    parser.add_argument("--pyramid_min_size", "-pms", help="The minimum pyramid size in pixels", type=int, default=DEFAULT_MIN_PYRAMID_SIZE)
    parser.add_argument("--pyramid_scale_factor", "-psf", help="The scale factor to scale images in the pyramid", type=float, default=DEFAULT_PYRAMID_SCALE_FACTOR)
    parser.add_argument("--bounding_box", "-bb", help="The sub-section of the geotif to analyze", default=None)
    parser.add_argument("--gpu_flag", "-gf", help="The flag to set when using gpu", default=DEFAULT_GPU_FLAG)
    parser.add_argument("--image_name", "-in", help="The name of image to include in name field of output vectors", default=None)
    parser.add_argument("--pyramid_window_sizes", "-pws", help="Sliding window sizes", default=None)
    parser.add_argument("--pyramid_step_sizes", "-pss", help="Sliding window step sizes", default=None)
    parser.add_argument("--bands", "-b", help="Band", default=None)
    parser.add_argument("--num_processes", "-np", help="Number of CPU processes", default=DEFAULT_NUM_PROCESSES)
    
    parser.add_argument("--log_level", "-ll", 
        help="Level of logging. (default error)", default="error", 
        choices=["error", "warn", "debug", "info"])

    parser.add_argument("--class_thresholds", "-ct",  
        help="Class specific thresholds", default=None, nargs='+')
    
    parser.add_argument("--output_vectors_dir", "-vd", help="Output vector path", default=OUTPUT_VECTORS_DIR_PATH)
    parser.add_argument("--output_vectors_zip_path","-vzp",help="Output vector zip file path", default=OUTPUT_VECTORS_ZIP_PATH)
    parser.add_argument("--zip_vectors", "-zv", help="Flag to zip vector json or not", default= "True")
    args = parser.parse_args(argv)
    args = validate_args(args)
    return args


def validate_args(args):
    validate_file("tif", args.tif, False, ".tif")
    validate_file("imd", args.imd, False, ".imd")
    validate_files("model_paths", args.model_paths, True)
    if args.threshold < 0.0:
        error_msg = "The provided threshold of %s is below 0" % args.threshold
        log.error(error_msg)
        raise ValueError(error_msg)

    validate_num_above_0("win_size", args.win_size)
    validate_num_above_0("step_size", args.step_size)
    validate_num_above_0("pyramid_min_size", args.pyramid_min_size)
    validate_num_above_value("pyramid_scale_factor", args.pyramid_scale_factor, 1)

    # ensure the min pyramid size < win_size
    if args.win_size < args.pyramid_min_size:
        error_msg = "The provided min pyramid size (%s) is greater than the window size (%s)" % (args.pyramid_min_size, args.win_size)
        log.error(error_msg)
        raise ValueError(error_msg)
    if args.bounding_box:
        validate_bbox(args.bounding_box)

    # Validate and modify to create a list of integers [150, 100, etc.]
    args.pyramid_window_sizes = validate_create_int_list("pyramid_window_sizes", args.pyramid_window_sizes)
    args.pyramid_step_sizes = validate_create_int_list("pyramid_step_sizes", args.pyramid_step_sizes)

    # Check pyramid_window_sizes and pyramid_step_sizes has the same length
    if  args.pyramid_window_sizes and args.pyramid_step_sizes and len(args.pyramid_step_sizes) != len(args.pyramid_window_sizes):
        raise RuntimeError("Pyramid window sizes length {} != Pyramid step sizes length {} ".format(len(args.pyramid_window_sizes), len(args.pyramid_step_sizes)))

    args.bands = validate_create_int_list("bands", args.bands)

    # Cast as integer
    args.num_processes = int(args.num_processes)
    validate_num_above_0("num_processes", args.num_processes)

    if args.gpu_flag.lower() == 'true' and args.num_processes > 1:
        log.info('WARNING: Setting gpu_flag to True requires the use of only one process. Setting num_processes = 1.')
        args.num_processes = 1

    # Now set the log level (already guaranteed to be one of the following)
    if args.log_level.lower() == "error":
        log_level_val = log.ERROR
    elif args.log_level.lower() == "warn":
        log_level_val = log.WARN
    elif args.log_level == "debug":
        log_level_val = log.DEBUG
    elif args.log_level == 'info':
        log_level_val = log.INFO
    log.getLogger().setLevel(log_level_val)

    # Now set the thresholds
    if args.class_thresholds is None:
       args.class_thresholds = {}
    else:
        try:
            args.class_thresholds = class_thresholds_dict(args.class_thresholds)
        except Exception, e: 
            log.info('ERROR: Error setting class thresholds. Format should be Class Name:Value .')
            raise RuntimeError(str(e))

    return args

def class_thresholds_dict(thresholds):
    class_thresh_dict = {}
    tmp_key_list = []
    for item in thresholds:
        if ':' in item:
            item_split = item.split(':')
            tmp_key_list.append(item_split[0])
            threshold = float(item_split[1])
            class_name = ""
            for key_item in tmp_key_list:
                class_name += key_item + " "
            # Add to dicitonary, remove empty space at the end
            class_thresh_dict[class_name[:-1].lower()] = threshold
            tmp_key_list = []
        else:
            tmp_key_list.append(item)
    return class_thresh_dict


def validate_files(arg_name, file_paths, is_dir, extension=None):
    if file_paths is None:
        error_msg = "The path provided for %s is None" % arg_name
        log.error(error_msg)
        raise ValueError
    for i in range(len(file_paths)):
        validate_file("%s[%s]" % (arg_name, i), file_paths[i], is_dir, extension)


def validate_file(arg_name, file_path, is_dir, extension=None):
    if file_path is None:
        error_msg = "The path provided for %s is None" % arg_name
        log.error(error_msg)
        raise ValueError
    if not os.path.exists(file_path):
        error_msg = "The path provided for %s (%s) does not exist" % (arg_name, file_path)
        log.error(error_msg)
        raise ValueError(error_msg)
    if extension and not file_path.lower().endswith(extension):
        error_msg = "The path provided for %s (%s) does not end with the extension of %s" % (arg_name, file_path, extension)
        log.error(error_msg)
        raise ValueError(error_msg)
    if not is_dir and not os.path.isfile(file_path):
        error_msg = "The path provided for %s (%s) is not a file but is expected to be one" % (arg_name, file_path)
        log.error(error_msg)
        raise ValueError(error_msg)
    if is_dir and not os.path.isdir(file_path):
        error_msg = "The provided for %s (%s) is not a directory but is expected to be one" % (arg_name, file_path)
        log.error(error_msg)
        raise ValueError(error_msg)


def validate_num_above_value(arg_name, actual, expected):
    if actual < expected:
        error_msg = "The provided value (%s) for %s is below %s" % (actual, arg_name, expected)
        log.error(error_msg)
        raise RuntimeError("The provided value (%s) for %s is below %s" % (actual, arg_name, expected))


def validate_create_int_list(arg_name, value):
    if value is None:
        return 

    error_msg = "The provided value %s for %s should be a list of integers" % (value, arg_name)
    val_list = list(ast.literal_eval(value))

    for index, val in enumerate(val_list):
        try: 
            val_list[index] = int(val)
        except:
            log.error(error_msg)
            raise RuntimeError(error_msg)

    return val_list

def validate_num_above_0(arg_name, value):
    validate_num_above_value(arg_name, value, 0)


def validate_bbox(bounding_box):
    bounding_box_values = [float(val.strip()) for val in bounding_box.split(" ")]
    if len(bounding_box_values) != 4:
        raise RuntimeError("The provided value (%s) for %s must have only 4 values" % (bounding_box, "bounding_box"))
    if bounding_box_values[0] < -180:
        raise RuntimeError("The provided value (%s) for minX must be more than -180" % bounding_box_values[0])
    if bounding_box_values[1] < -90:
        raise RuntimeError("The provided value (%s) for minY must be more than -90" % bounding_box_values[1])
    if bounding_box_values[2] > 180:
        raise RuntimeError("The provided value (%s) for maxX must be less than 180" % bounding_box_values[2])
    if bounding_box_values[3] > 90:
        raise RuntimeError("The provided value (%s) for maxY must be less than 90" % bounding_box_values[3])

def parse_imd_file(imd_file_path):
    return "sat_id", "cat_id", "date"


def get_classifier_paths(model_dir):
    caffemodel = None
    deploy_file = None
    mean_file = None
    labels_file = None
    for root, dirs, files in os.walk(model_dir):
        for model_file in files:
            full_path = os.path.join(root, model_file)
            if model_file.endswith(CAFFEMODEL_SUFFIX):
                caffemodel = full_path
            elif model_file.endswith(DEPLOY_FILE_SUFFIX):
                deploy_file = full_path
            elif model_file.endswith(MEAN_FILE_SUFFIX):
                mean_file = full_path
            elif model_file.endswith(LABELS_FILE_SUFFIX):
                labels_file = full_path
    return caffemodel, deploy_file, mean_file, labels_file

def calc_tile_ranges(num_tiles, num_procs):
    tiles_per_proc = num_tiles/num_procs
    tile_ranges = np.zeros((num_procs+1), dtype=np.int)
    for i in range(num_procs):
        if i < num_tiles % num_procs:
            # First procs divide remainder (if it exists)
            tile_ranges[i+1] = tile_ranges[i] + tiles_per_proc + 1
        else:
            tile_ranges[i+1] = tile_ranges[i] + tiles_per_proc
    return tile_ranges

def stretchData(data,pct=2):
    # Linear 2pct stretch 
    a,b = np.percentile(data, [pct, 100-pct])
    return 255 * ( data - a ) / (b-a+0.001)

def tile_list_classifier(tile_list, args, image_name, item_date, sat_id, cat_id, mean_files, caffemodels, deploy_files, labels_files, bands, threshold_dict):

    # Create classifier object
    classifier = CaffeBatchClassifier(
                     caffe_models=caffemodels,
                     deploy_files=deploy_files, 
                     label_files=labels_files, 
                     mean_files=mean_files,
                     gpu_flag=args.gpu_flag)
    
    # open image
    gdal_image = GDALImage(args.tif, RASTER_DIM, RASTER_DIM, strideX=RASTER_STRIDE, strideY=RASTER_STRIDE, bands=bands, padWithZeros=True) 

    for tile in tile_list:
        
        timerStart = time.time()
        
        image = gdal_image.readTile( tile[0], tile[1], tile[2], tile[3] )
                
        if image.shape[0] < classifier.get_caffe_num_channels():
            log.info("Exception: Cannot run imagery with fewer bands than Caffe model.")
            raise RuntimeError
                                                
        classify_broad_area_multi_process(gdal_image, image, tile[0], tile[1], args,
                                          image_name, item_date, sat_id, cat_id,
                                          mean_files, caffemodels, deploy_files, labels_files,
                                          classifier, args.gpu_flag, threshold_dict)

        timerEnd = time.time()
        log.info("Time for tile: "+str(timerEnd-timerStart))

def set_class_thresholds(labels_files, global_threshold, class_threshold_dict):
    # Assume all models have the same labels file
    label_file_name = labels_files[0]
    thresh_dict = {}
    with open(label_file_name, 'r') as label_file:
        for line in label_file:
            thresh_dict[line[:-1].lower()] = global_threshold

    for k, v in class_threshold_dict.iteritems():
        if k in thresh_dict:
            thresh_dict[k.lower()] = v
        else:
            log.info("ERROR: Class name {} not found in model label file {}.".format(k, label_file_name))
            raise RuntimeError

    return thresh_dict

#@profile(filename="ens_gpu.prof", profiler="cProfile")
def process_args(args):
    time0 = time.time()
    image_name = args.image_name or args.tif

    caffemodels = []
    deploy_files = []
    mean_files = []
    labels_files = []
    for i in range(len(args.model_paths)):
        caffemodel, deploy_file, mean_file, labels_file = get_classifier_paths(args.model_paths[i])
        caffemodels.append(caffemodel)
        deploy_files.append(deploy_file)
        mean_files.append(mean_file)
        labels_files.append(labels_file)

    if args.class_thresholds == None:
        args.class_thresholds = {}
        
    threshold_dict = set_class_thresholds(labels_files, args.threshold, args.class_thresholds)
    log.info("Thresholds: {}".format(threshold_dict))

    init_vector_dir(args.output_vectors_dir)

    # Open this once here to generate the list of tiles to distribute
    gdal_image = GDALImage(args.tif, RASTER_DIM, RASTER_DIM, strideX=RASTER_STRIDE, strideY=RASTER_STRIDE, padWithZeros=True)
    
    # Parse metadata if available
    item_date, cat_id, sat_id = ( None, None, None)
    if args.imd:
        try:
            item_date, cat_id, sat_id = parse_imd(args.imd)
        except Exception, e:
            log.info("Could not parse imd file "+str(e))

    # Set bounding box before tile generation
    if args.bounding_box:
        bb = [float(val.strip()) for val in args.bounding_box.split(" ")]
        gdal_image.setGeoBoundingBox(bb[0],bb[1],bb[2],bb[3])        

    # Return the list of tiles
    all_tiles = list( gdal_image.nextTile() )

    # Calculate the number of tiles
    num_tiles = len(all_tiles)

    num_procs = args.num_processes

    proc_tile_ranges = calc_tile_ranges(num_tiles, num_procs)
    log.info("Time to generate {} tiles {} seconds".format(num_tiles, time.time() - time0))

    time0 = time.time()
    
    if ( args.num_processes < 2 ):
        tile_list_classifier(all_tiles, args, image_name, item_date, sat_id, cat_id, mean_files,
                             caffemodels, deploy_files, labels_files, args.bands, threshold_dict)
    else:
        manager = mp.Manager()
        pool = mp.Pool(processes=args.num_processes)
        log.warn("pool size = %s" % str(args.num_processes))
     
        # For each process launch one async job
        for i in range(num_procs):
            tile_list = all_tiles[proc_tile_ranges[i]:proc_tile_ranges[i+1]]
            # Each process needs to instantiate its own gdal image due to native code dependencies, file handles etc., 
            # it cannot be passed in to apply_async         
            pool.apply_async(tile_list_classifier,
                (tile_list, args, image_name, item_date, sat_id, cat_id, mean_files,
                 caffemodels, deploy_files, labels_files, args.bands, threshold_dict))
     
        pool.close()
        pool.join()

    log.info("Total detection time {} seconds".format(time.time() - time0))
    
    # close geojson results
    with open( os.path.join( args.output_vectors_dir, OUTPUT_VECTORS_FILE ), "a") as fout:
        fout.write("] }")

    time0 = time.time()
    if args.zip_vectors != None and args.zip_vectors.lower() == "true":
        zip_file_cnt = len(glob.glob(os.path.join(args.output_vectors_dir,"*.json")))
        log.info("Start to zip {0} json files".format(zip_file_cnt))
        zip_vectors_file(args.output_vectors_dir, args.output_vectors_zip_path)
        log.info("Time to zip {} files in {} seconds".format(zip_file_cnt, time.time()-time0))
    
def write_status_file(file_path, exception=None):
    status_file = open(file_path, 'w')
    if exception:
        status_dict['status'] = STATUS_FAILED
        status_dict['reason'] = str(exception)
    json.dump(status_dict, status_file)
    status_file.close()

def parse_imd(filename):
    date = None
    catId = None
    satId = None
    with open(filename,"r") as fin:
        for line in fin:
            m = re.match(".*firstLineTime = (.+)\;", line )
            if m: date = m.group(1)
            m = re.match('.*satId = "(.+)"\;', line )    
            if m: satId = m.group(1)
            m = re.match('.*CatId = "(.+)"\;', line )    
            if m: catId = m.group(1)
    return ( date, catId, satId )
    
def main(argv):
    log.getLogger().setLevel(log.INFO)
    args = None
                
    try:
        args = parse_args(argv)
        process_args(args)
        write_status_file(args.status_path)
    except Exception, e:
        log.error("Exception:"+str(e))
 
        if args and args.status_path:
            write_status_file(args.status_path, traceback.format_exc())
        else:
            write_status_file(DEFAULT_STATUS_JSON_PATH, traceback.format_exc())
 
        raise

if __name__ == '__main__':
    main(sys.argv[1:])

