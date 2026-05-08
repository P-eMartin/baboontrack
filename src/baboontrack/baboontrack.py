import sys
import os
import cv2
import numpy as np
import time
import datetime
import copy
import torch
from argparse import ArgumentParser
import pdb
from scipy.ndimage import gaussian_filter
import pdb

# Megadetector
from megadetector.detection import run_detector

# Print Versions
print('Python version: %s' % (sys.version))
print('OpenCV version: %s' % (cv2.__version__))
print('PyTorch version: %s' % (torch.__version__))

# Utility functions
from .bt_utils.io_utils import print_and_log, setup_logger, close_log, progress_bar, save_dict_as_csv, zip_folder, get_value_with_precision
from .bt_utils.json_utils import save_json_file, load_json_file
from .bt_utils.img_utils import VideoFrameIterator

# Help variables
from .help import *

# GUI
from .gui import run_with_gui, check_gui_stop

def get_last_tracks(all_tracks):
    '''
    Get the last tracks from the tracking buffer.

    Args:
        all_tracks: list of list of dict, the tracking buffer containing the tracks for each frame

    Returns:
        list of dict, the last tracks from the tracking buffer
    '''
    unique_tracks = []
    track_ids = []
    for detections in reversed(all_tracks):
        for detection in detections:
            if detection['track_id'] not in track_ids:
                track_ids.append(detection['track_id'])
                unique_tracks.append(detection)
    return unique_tracks

def compute_iou(boxA, boxB, image_size):
    '''
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA: list of int, the first bounding box in the format [x, y, w, h]
        boxB: list of int, the second bounding box in the format [x, y, w, h]
        image_size: list of int, the size of the image in the format [width, height]

    Returns:
        float, the IoU value
    '''
    xA = max(boxA[0], boxB[0])*image_size[0]
    yA = max(boxA[1], boxB[1])*image_size[1]
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])*image_size[0]
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])*image_size[1]
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]* image_size[0] * image_size[1]
    boxBArea = boxB[2] * boxB[3]* image_size[0] * image_size[1]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_detections(detections, last_tracks, track_id, image_size, score, iou_threshold=0.3, default_class_id=None, log=None):
    '''
    Process each detection and assign a track ID based on the last tracks and IoU threshold.
    Visibility is set to 1 if no overlap with other tracks, and decreases as the IoU with other tracks increases.

    Args:
        detections: list of dict, the detections for the current frame
        last_tracks: list of dict, the last tracks from the tracking buffer
        track_id: int, the current track ID
        image_size: list of int, the size of the image in the format [width, height]
        score: float, the minimum detection score
        iou_threshold: float, the IoU threshold for matching detections with tracks
        default_class_id: int, the default class ID for detections without a matching track
        classes: list of str, the list of class names to consider for tracking (default ["animal","person","vehicle"])
    Returns:
        int, the updated track ID after assignment
    '''
    to_delete = []
    for idx, detection in enumerate(detections):
        # Remove detections with low score
        if detection['conf'] < score:
            print_and_log('Detection with low score found (frame %d): %f' % (idx, detection['conf']), log=log)
            to_delete.append(idx)
            continue
        # Assign track ID based on IoU with last tracks
        best_iou = 0
        vis = 1
        best_track_id = None
        for track in last_tracks:
            iou = compute_iou(detection['bbox'], track['bbox'], image_size)
            if iou > best_iou:
                vis = 1 - best_iou
                best_iou = iou
                best_track_id = track['track_id']
        if best_iou > iou_threshold:
            detection['track_id'] = best_track_id
        else:
            detection['track_id'] = track_id
            track_id += 1
        if default_class_id is not None:
            detection['id'] = default_class_id
            detection['id_score'] = 0
        detection['det'] = int(detection['category'])-1
        detection['det_score'] = detection['conf']
        detection['visibility'] = get_value_with_precision(vis)
        # Remove the normalized keys
        del detection['category']
        del detection['conf']

    # Remove detections with low score
    for idx in reversed(to_delete):
        del detections[idx]

    return track_id

def save_mot_format(detection_dict, output_path, image_size=None, labels=None):
    '''
    Save the detection and tracking results in MOT format.

    Args:
        detection_dict: dict, the detection and tracking results
        output_path: str, the path to save the output files
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        labels: list of str, the list of labels to save in a separate file (default None)

    Returns:
        int, 1 if the file was properly saved
    '''
    # Create the MOT format dictionary - leading # in the first key so it is considered as comment in the MOT format
    folder_to_zip = os.path.join(output_path, 'gt')
    os.makedirs(folder_to_zip, exist_ok=True)
    mot_dict = {
        'frame_id': [],
        'track_id': [],
        'x': [],
        'y': [],
        'w': [],
        'h': [],
        'not ignored': [],
        'class_id': [],
        'visibility': [],
        'skipped': []
    }
    # Loop over the detection_dict and fill the MOT format dictionary
    for idx, dets in enumerate(detection_dict):
        for det in dets:
            mot_dict['frame_id'].append(idx+1)
            mot_dict['track_id'].append(det['track_id']+1)
            mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
            mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
            mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
            mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
            mot_dict['not ignored'].append(1)
            mot_dict['class_id'].append(int(det['id']+1))
            mot_dict['visibility'].append(det['visibility'])
            mot_dict['skipped'].append(0)
    save_dict_as_csv(mot_dict, os.path.join(folder_to_zip, 'gt.txt'), without_headers=True)
    # Save labels if provided
    if labels is not None:
        with open(os.path.join(folder_to_zip, 'labels.txt'), 'w') as f:
            for label in labels:
                f.write('%s\n' % (label))
    zip_folder(folder_to_zip, os.path.join(output_path, 'mot.zip'))
    return 1

def detect(my_video, output_path, device='cpu', tracking_size=30, score=0.5, display_fct=None, model_path='md_v5b.0.0.pt', log=None):
    '''
    Detect the Baboons in the video using Megadetector and track them.

    Args:
        my_video: VideoFrameIterator, the video to process
        output_path: str, the path to save the output file
        device: str, the device to use for the detection
        tracking_size: int, the size of the tracking buffer in number of frames (default 30)
        score: float, the minimum detection score (default 0.5)
        log: logger, the logger to print the information (default None)
        display_fct: function, the function to display the results in real-time (default None)

    Returns:
        dict, the detection and tracking results
    '''
    # Initialization
    
    ## Check if the output file already exists
    output_file = os.path.join(output_path, 'results.json')
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Skipping detection and tracking.' % (output_file), log=log)
        return output_file

    ## Variables
    track_id = 0
    det_results = []
    tracking_buffer = []
    last_tracks = []
    start_time = time.time()

    ## Megadetector
    if model_path is not None:
        if os.path.exists(model_path):
            print_and_log('Loading Megadetector model from %s' % (model_path), log=log)
        else:
            print_and_log('Megadetector model path %s does not exist. Loading default model.' % (model_path), log=log)
            model_path = 'MDV5B'
    else:
        model_path = 'MDV5B'
    model = run_detector.load_detector(
        model_path,
        detector_options={'device':device}
    ) # 1452MB on gpu
    det_classes = ["animal","person","vehicle"]

    # Loop over the video frames
    print_and_log('Processing video %s' % (my_video.path), log=log)
    for idx, frame in enumerate(my_video):
        if idx == 0:
            image_size = frame.shape[:2][::-1]
            print_and_log('Video resolution: %s' % (str(image_size)), log=log)

        ## Progress bar with estimated time remaining
        elapsed_time = time.time() - start_time
        progress_bar(
            idx,
            len(my_video),
            'Detection and Tracking Progress with currently %d tracks.%s' % (
                track_id,
                '(%ds left)' % (elapsed_time/idx*(len(my_video)-idx-1)) if idx else ''
            ),
            log=log
        )
        
        ## Detection (Could be improved by running the detection on a batch of frames instead of one by one)
        det_result = model.generate_detections_one_image(frame, detection_threshold=score)['detections'] # image_id=idx

        ## Assign track_ids
        track_id = process_detections(det_result, last_tracks, track_id, image_size, score, iou_threshold=0.3, default_class_id=0)

        ## Update Tracking
        if len(tracking_buffer) > tracking_size:
            tracking_buffer.pop(0)
        tracking_buffer.append(det_result)
        last_tracks = get_last_tracks(tracking_buffer)

        ## Save and display results
        det_results.append(det_result)
        if display_fct is not None:
            display_fct(frame, det_result)
    progress_bar(len(my_video), len(my_video), 'Detection and Tracking done in %ds with %d tracks' % (time.time() - start_time, track_id), log=log, completed=True)
        
    # Saving
    output_results = {'detections': det_results, 'detection_classes': det_classes, 'format': 'xywh', 'image_size': image_size}
    save_json_file(output_results, output_file)
        
    return output_results

def classify(detection_dict, my_video, output_path, log=None):
    '''
    Classify the tracks of the detected Baboons using a pre-trained classifier and a dictionary with extracted features from the tracks.

    Args:
        detection_dict: dict, the detection and tracking results
        my_video: VideoFrameIterator, the video to process
        output_path: str, the path to save the output file
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the classification results
    '''
    # Initialization
    ## Check if the output file already exists
    output_file = os.path.join(output_path, 'classification_results.json')
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Loading existing file.' % (output_file), log=log)
        return load_json_file(output_file)
    ## Classification names
    classes = ["NoID"]
    
    ## Check if detection_dict is filepath or dict
    if isinstance(detection_dict, str):
        if not os.path.exists(detection_dict):
            print_and_log('Error: detection_dict %s must be a file or a dict' % (detection_dict), log=log, to_print=False)
            raise ValueError('No detection_dict provided.')
        print_and_log('Loading detection_dict from %s' % (detection_dict), log=log)
        detection_dict = load_json_file(detection_dict)
    
    classification_dict = copy.deepcopy(detection_dict)
    # TODO: implement the classification of the tracks using a pre-trained classifier and a dictionary with extracted features from the tracks.

    # Saving
    classification_dict['classification_classes'] = classes
    save_json_file(classification_dict, output_file)

    return classification_dict

def main(args, log=None):
    '''
    Main function to process the video.
    Perform first the detection and tracking of the Baboons.
    Then it classify each track using a Baboon dictionary with extracted features from the tracks and a pre-trained classifier.
    Optionally, it creates a visualization of the tracks and the classification results on the video.

    Args:
        args: argparse.Namespace, the arguments
        log: logger, the logger to print the information

    Returns:
        int, 1 if the function ran successfully
    '''
    # Check if there is an input
    if not os.path.exists(args.input_video):
        print_and_log('Error: input %s must be a file or a folder' % (args.input_video), log=log, to_print=False)
        raise ValueError('No input provided.')
    
    # Chrono
    start_time = time.time()

    # Video object initialization
    my_video = VideoFrameIterator(args.input_video, log=log)
    # my_video.check_video()

    # Detection and tracking
    if check_gui_stop(log=log): return 0
    detection_dict = detect(
        my_video,
        args.output,
        device=args.device,
        score=args.det_score,
        tracking_size=args.tracking_size,
        log=log,
        display_fct=args.display_fct
    )
    if check_gui_stop(log=log): return 0

    classes = ["NoID"]

    # Classification
    classification_dict = classify(
        detection_dict,
        my_video,
        args.output,
        log=log
    )
    if check_gui_stop(log=log): return 0

    # Save MOT format
    save_mot_format(
        classification_dict['detections'],
        os.path.join(args.output, 'mot'),
        image_size=classification_dict['image_size'],
        labels=classes
    )

    # Create the video
    if args.video_demo:
        my_video.reset_video()
        my_video.plot_annotations(
            classification_dict['detections'],
            os.path.join(args.output, 'video_demo.mp4'),
            max_res=args.max_res,
            display_fct=args.display_fct,
            detection_classes=classification_dict['detection_classes'],
            classification_classes=classification_dict['classification_classes'],
            del_imgs=args.del_imgs,
            log=log
        )
        if check_gui_stop(log=log): return 0

    print_and_log("Processing of %s finished in %ds." % (args.input_video, time.time()-start_time), log=log)
    close_log(log)


def split_or_empty(string):
    '''
    Split string and return the last element if not empty, else return empty string.

    Args:
        string: string to split.

    Returns:
        string, last element of the split string if not empty, else empty string.
    '''
    return string if string == '' else os.path.splitext(os.path.basename(os.path.normpath(string)))[0]

def infer_args_name(args):
    '''
    Infer the output folders from the arguments if not provided.
    args is modified in place.
    
    Args:
        args: argparse.Namespace, the arguments
    
    Returns:
        int, 1 if the function ran successfully
    '''

    name_input = split_or_empty(args.input_video)

    # Check output folder
    if args.output == '':
        args.output = os.path.join('output','%s%s' % (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
                ('_i_%s' % (name_input)) if name_input != '' else ''
            )
        )
    return 1

def get_args():
    '''
    Get the arguments from the command line, process them and return them.

    Returns:
        args: argparse.Namespace, the arguments
    '''
    parser = ArgumentParser(description="Process thermal images to compute landmarks and signals.")
    parser.add_argument(
        '-i', '--input_video',
        default='',
        type=str,
        help=helptext_input_video
    )
    parser.add_argument(
        '-o', '--output',
        default='',
        type=str,
        help=helptext_output
    )
    parser.add_argument(
        '-v', '--video_demo',
        action='store_true',
        help=helptext_video_demo
    )
    parser.add_argument(
        '-r', '--max_res',
        default=1080,
        type=int,
        help=helptext_max_res
    )
    parser.add_argument(
        '-s', '--det_score',
        default=0.5,
        type=float,
        help=helptext_det_score
    )
    parser.add_argument(
        '-D', '--del-imgs',
        action='store_true',
        help=helptext_del_imgs
    )
    parser.add_argument(
        '-d', '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help=helptext_device
    )
    parser.add_argument(
        '-t', '--tracking_size',
        type=int,
        default=30,
        help=helptext_tracking_size
    )
    parser.add_argument(
        '-g', '--gui',
        action='store_true',
        help=helptext_gui
    )
    args = parser.parse_args()
    args.parser = parser
    infer_args_name(args)
    return args

def check_args(**kwargs):
    '''
    Check arguments and return them.

    Returns:
        args: argparse.Namespace, the arguments
    '''
    args = get_args()
    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            args.parser.print_help()
            raise ValueError('Attribute %s not found in args.' % (key))
    args.display_fct = None
    return args

def run(**kwargs):
    '''
    Run BaboonTrack with arguments.

    Args:
        kwargs: dict, the arguments

    Returns:
        int, 1 if BaboonTrack finished
    '''
    # Check arguments
    args = check_args(**kwargs)
    
    # Send arguments to main
    if args.gui:
        # Use GUI
        run_with_gui(args, main, check_args_fct=infer_args_name)
    else:
        os.makedirs(args.output, exist_ok=True)
        log = setup_logger(log_file=os.path.join(args.output, '%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))))
        print_and_log('Starting BaboonTrack without GUI with arguments: %s' % (args), log=log)
        main(args, log=log)
        close_log(log)
    
    # Finish
    print('BaboonTrack finished.')
    return 1

if __name__ == '__main__':
    # Run BaboonTrack
    run()

    # Exit
    sys.exit(0)
