import os
import json
import logging as log

import caffe_detector

GBDXRUNTIMEFILE='/mnt/work/gbdx_runtime.json'
PORTSFILE = '/mnt/work/input/ports.json'
STATUSFILE = '/mnt/work/status.json'
INPUTDIR = '/mnt/work/input'
OUTPUTDIR = '/mnt/work/output'
TEMPDIR = '/mnt/work/output/temp'

DEFAULT_LOGGING_DIR = OUTPUTDIR+'/log'
DEFAULT_OUTPUT_DIR = OUTPUTDIR+'/result'

# set up logging
if not os.path.exists(DEFAULT_LOGGING_DIR):
    os.makedirs(DEFAULT_LOGGING_DIR)
log.getLogger().addHandler( log.FileHandler(DEFAULT_LOGGING_DIR+'/app.log') )
#log.getLogger().addHandler( log.StreamHandler() )
log.getLogger().setLevel( log.WARNING )
# set up output dir and temp dir
if not os.path.exists(DEFAULT_OUTPUT_DIR):
    os.makedirs(DEFAULT_OUTPUT_DIR)
if not os.path.exists(TEMPDIR):
    os.makedirs(TEMPDIR)

def create_status_file(status, message):
    content = { 'status': status,
                'reason': message }
    with open(STATUSFILE, 'w') as f:
        f.write( json.dumps(content) )

def parse_ports_file(ports_file):
    with open(ports_file, 'r') as json_file:
        args_json = json.load(json_file)
    return args_json

def find_file(path,suffix):
    matches = []
    for dirName, subdirList, fileList in os.walk(path):
        for fname in fileList:
            if ( fname.lower().endswith(suffix) ):
                matches.append( dirName+os.sep+fname )
    return matches

def find_dir(path,suffix):
    matches = []
    for dirName, subdirList, fileList in os.walk(path):
        for fname in subdirList:
            if ( fname.lower().endswith(suffix) ):
                matches.append( dirName+os.sep+fname )
    return matches

def create_launch_arguments():
    args = []
    ports = parse_ports_file( PORTSFILE )
    
    images = find_file(INPUTDIR+os.sep+"image", ".tif")
    args += ['--tif',images[0]] 
    imdfile = find_file(INPUTDIR+os.sep+"image", ".imd")
    if len(imdfile) > 0:
        args += ["--imd",imdfile[0]]

    modeldir = INPUTDIR+os.sep+"model"
    args += ['--model_paths', modeldir ]
    if ports.has_key('threshold'):
        args += [ '--threshold', ports['threshold'] ]
    if ports.has_key('bounding_box'):
        args += [ '--bounding_box' , ports['bounding_box'] ]
    if ports.has_key('gpu_flag') and ports['gpu_flag'].lower() == 'true':
        args += [ '--gpu_flag', 'true']
    if ports.has_key('pyramid_window_sizes'):
        args += [ '--pyramid_window_sizes' , ports['pyramid_window_sizes' ]]
    if ports.has_key('pyramid_step_sizes'):
        args += [ '--pyramid_step_sizes' , ports['pyramid_step_sizes' ]]
    if ports.has_key('num_processes'):
        args += [ '--num_processes' , ports['num_processes' ]]
    if ports.has_key('class_thresholds'):
        args += [ '--class_thresholds']
        args += ports['class_thresholds' ].split()
    if ports.has_key('log_level'):
        args += [ '--log_level' , ports['log_level' ]]
                
    return args
    
if __name__ == '__main__':
    log.getLogger().setLevel(log.INFO)
    log.info('Application start')
    
    # parse ports file and build arguments
    launcher_args = create_launch_arguments()
    try:
        args = caffe_detector.parse_args(launcher_args)
        caffe_detector.process_args(args)
    except Exception, e:
        log.error("Exception:"+str(e))
        create_status_file('success','Caffe detection run failed. See logs for details')
        exit(1)
    
    # Write status file
    create_status_file('success','Caffe detection run succeeded.')
    log.info('Application end')
