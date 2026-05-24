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

# Megadetector
from megadetector.detection import run_detector

# Print Versions
print('Python version: %s' % (sys.version))
print('OpenCV version: %s' % (cv2.__version__))
print('PyTorch version: %s' % (torch.__version__))

# Utility functions
from .bt_utils.io_utils import print_and_log, setup_logger, close_log, progress_bar, get_value_with_precision
from .bt_utils.json_utils import save_json_file, load_json_file
from .bt_utils.img_utils import VideoFrameIterator
from .bt_utils.tracking import ReIDModel, init_tracker, update_tracker
from .bt_utils.eval_utils import evaluate_detection, evaluate_tracking, mot_gt_to_coco_gt, save_mot_format, save_coco_format, load_mot_format, mot_to_coco_format

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
    matches = []
    for idx, detection in enumerate(detections):
        # Remove detections with low score
        if detection['conf'] < score:
            print_and_log('Detection with low score found (frame %d): %f' % (idx, detection['conf']), log=log)
            to_delete.append(idx)
            continue
        # Match track ID based on IoU with last tracks
        best_iou = 0
        for track in last_tracks:
            iou = compute_iou(detection['bbox'], track['bbox'], image_size)
            if iou > best_iou:
                matches.append({'det_idx':idx, 'track_id': track['track_id'], 'iou': iou})
                best_iou = iou

    # Sort by best IoU first
    matches = sorted(matches, key=lambda x: x['iou'], reverse=True)

    assigned_dets = set()
    assigned_tracks = set()

    # Assign globally best matches
    for match in matches:
        det_idx = match['det_idx']
        track_id_match = match['track_id']

        if det_idx in assigned_dets or track_id_match in assigned_tracks:
            continue

        detections[det_idx]['track_id'] = track_id_match
        # Visibility
        detections[det_idx]['visibility'] = get_value_with_precision(1 - match['iou'])

        assigned_dets.add(det_idx)
        assigned_tracks.add(track_id_match)

    # Assign new track IDs to unmatched detections
    for idx, detection in enumerate(detections):

        if idx in to_delete:
            continue

        if 'track_id' not in detection:
            detection['track_id'] = track_id
            track_id += 1

        if default_class_id is not None:
            detection['id'] = default_class_id
            detection['id_score'] = 0
        
        detection['det'] = int(detection['category'])-1
        detection['det_score'] = detection['conf']
        detection['visibility'] = get_value_with_precision(1-max([match['iou'] for match in matches if match['det_idx'] == idx and match['track_id'] != detection['track_id']], default=0))
        # Remove the normalized keys
        del detection['category']
        del detection['conf']

    # Remove detections with low score
    for idx in reversed(to_delete):
        del detections[idx]

    return track_id


def detect(my_video, output_path, device='cpu', tracking_size=60, score=0.5, display_fct=None, det_model='md_v5b.0.0.pt', log=None):
    '''
    Detect the Baboons in the video using Megadetector and track them.

    Args:
        my_video: VideoFrameIterator, the video to process
        output_path: str, the path to save the output file
        device: str, the device to use for the detection
        tracking_size: int, the size of the tracking buffer in number of frames (default 60)
        score: float, the minimum detection score (default 0.5)
        log: logger, the logger to print the information (default None)
        display_fct: function, the function to display the results in real-time (default None)

    Returns:
        dict, the detection and tracking results
    '''
    # Initialization
    
    ## Check if the output file already exists
    output_file = os.path.join(output_path, 'detection_%s.json') % (os.path.basename(det_model).split('.')[0])
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
    if det_model is not None:
        if os.path.exists(det_model):
            print_and_log('Loading Megadetector model from %s' % (det_model), log=log)
        else:
            print_and_log('Megadetector model path %s does not exist. Loading default model.' % (det_model), log=log)
            det_model = 'MDV5B'
    else:
        det_model = 'MDV5B'
    model = run_detector.load_detector(
        det_model,
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

def track(my_video, detection_dict, output_path, device='cpu', tracking_size=60, score=0.5, log=None, tracker_type=None):
    '''
    Track the Baboons in the video using the detection results.

    Args:
        my_video: VideoFrameIterator, the video to process
        detection_dict: dict, the detection results
        output_path: str, the path to save the output file
        device: str, the device to use for the tracking
        tracking_size: int, the size of the tracking buffer in number of frames (default 30)
        score: float, the minimum detection score (default 0.5)
        log: logger, the logger to print the information (default None)
        tracker_type: str, the type of tracker to use (default None)

    Returns:
        dict, the tracking results
    '''
    # Initialization
    ## Check if the output file already exists
    output_file = os.path.join(output_path, 'tracking_%s.json' % (tracker_type if tracker_type else 'default'))
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Skipping tracking. Loading existing file.' % (output_file), log=log)
        return output_file
    
    ## Check if detection_dict is filepath or dict
    if isinstance(detection_dict, str):
        print_and_log('Loading detection_dict from %s' % (detection_dict), log=log)
        detection_dict = load_json_file(detection_dict)

    start_time = time.time()
    if tracker_type == "bytetrack":
        from .bt_utils.bytetrack.byte_tracker import BYTETracker
        print_and_log('Tracking with ByteTrack', log=log)
        feat_model = None
        tracker = BYTETracker()
    elif tracker_type == "deepsort":
        print_and_log('Tracking with DeepSort', log=log)
        ## Feature extractor
        feat_model = ReIDModel(device=device)
        ## Tracker
        tracker = init_tracker(max_cosine_distance=0.5, nn_budget=100, max_iou_distance=0.5, max_age=tracking_size, n_init=0)
    elif tracker_type == "botsort":
        print_and_log('Tracking with BoTSORT', log=log)
        from .bt_utils.bot_sort.bot_sort import BoTSORT
        feat_model = ReIDModel(device=device)
        tracker = BoTSORT(encoder=feat_model)
    else:
        # None - return the detection results without tracking
        print_and_log('No tracking, just returning detection results', log=log)
        return detection_dict

    # Loop over the video frames and detection results
    n_tracks = -1
    all_tracks = []
    for idx, (frame, det_result) in enumerate(zip(my_video, detection_dict['detections'])):
        ## Progress bar with estimated time remaining
        elapsed_time = time.time() - start_time
        progress_bar(
            idx,
            len(my_video),
            'Tracking Progress with currently %d tracks.%s' % (
                n_tracks,
                '(%ds left)' % (elapsed_time/idx*(len(my_video)-idx-1)) if idx else ''
            ),
            log=log
        )

        ## Update tracker with the detection results
        if tracker_type in ["bytetrack", "botsort"]:
            # Bboxes in (x1, y1, x2, y2) format pixel coordinates and scores
            bboxes = []
            for det in det_result:
                x, y, w, h = det['bbox']
                x1 = int(x * detection_dict['image_size'][0])
                y1 = int(y * detection_dict['image_size'][1])
                x2 = int((x + w) * detection_dict['image_size'][0])
                y2 = int((y + h) * detection_dict['image_size'][1])
                bboxes.append([x1, y1, x2, y2])
            bboxes = np.array(bboxes)
            scores = np.array([det['det_score'] for det in det_result])
            if tracker_type == "bytetrack":
                _current_tracks = tracker.update(bboxes, scores)
            elif tracker_type == "botsort":
                # Expecting bboxes in (x1, y1, x2, y2) format pixel coordinates, scores and classid (same classid for all detections here)
                bboxes_scores = np.hstack((bboxes, scores[:, np.newaxis], np.ones((len(scores), 1))))
                _current_tracks = tracker.update(bboxes_scores, frame)
            # Convert back to (x, y, w, h) format and normalized coordinates
            current_tracks = []
            for track in _current_tracks:
                x1, y1, w, h = track.tlwh
                tmp = {}
                tmp['bbox'] = [float(x1 / detection_dict['image_size'][0]), float(y1 / detection_dict['image_size'][1]), float(w / detection_dict['image_size'][0]), float(h / detection_dict['image_size'][1])]
                tmp['track_id'] = track.track_id
                tmp['det_score'] = float(track.score)
                n_tracks = max(n_tracks, track.track_id)
                current_tracks.append(tmp)
            all_tracks.append(current_tracks)
        else:
            current_tracks, max_track_id = update_tracker(tracker, frame, feat_model, det_result)
            n_tracks = max(n_tracks, max_track_id)
            all_tracks.append(current_tracks)
    progress_bar(len(my_video), len(my_video), 'Tracking done in %ds with %d tracks' % (time.time() - start_time, n_tracks), log=log, completed=True)

    # Saving
    output_results = {'detections': all_tracks, 'format': 'xywh', 'image_size': detection_dict['image_size']}
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
    classes = ['NoID']
    n_tracks = 0
    
    ## Check if detection_dict is filepath or dict
    if isinstance(detection_dict, str):
        print_and_log('Loading detection_dict from %s' % (detection_dict), log=log)
        detection_dict = load_json_file(detection_dict)
    
    classification_dict = copy.deepcopy(detection_dict)
    # TODO: implement the classification of the tracks using a pre-trained classifier and a dictionary with extracted features from the tracks.
    for dets_per_frame in classification_dict['detections']:
        # Just assign track_id as the class for now
        for det in dets_per_frame:
            track_id = det['track_id']
            det['det'] = track_id-1
            n_tracks = max(n_tracks, track_id)
            if 'det_score' not in det:
                det['det_score'] = None
            if 'id' not in det:
                det['id'] = 0
            det['id_score'] = None

    # Saving
    classification_dict['detection_classes'] = [f"Track {i}" for i in range(1, n_tracks + 1)]
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

    # Load ground truth if evaluation is enabled
    if args.eval_detection or args.eval_tracking or args.eval_classification:
        # With class ID being the track id but also the det ID
        gt_file_class_mot = os.path.join(os.path.dirname(args.input_video), 'frame-1639-2000-mot.zip')
        boundaries = [int(x) for x in os.path.basename(gt_file_class_mot).split('-')[1:3]]
        gt_dict_mot_cat = load_mot_format(gt_file_class_mot, boundaries=boundaries)

    # Detection
    ## TODO: Detection - tracking with SAM 3
    if check_gui_stop(log=log): return 0
    det_model='md_v5b.0.0.pt'
    detection_dict = detect(
        my_video,
        args.output,
        device=args.device,
        score=args.det_score,
        tracking_size=args.tracking_size,
        det_model=det_model,
        log=log,
        display_fct=args.display_fct
    )

    if args.eval_detection:
        # Load detection results if not already loaded
        if not isinstance(detection_dict, dict):
            print_and_log('Loading detection_dict from %s' % (detection_dict), log=log)
            detection_dict = load_json_file(detection_dict)
        # Convert MOT GT to COCO format for evaluation
        labels_detection = ['Baboon']
        gt_dict_coco_det = mot_gt_to_coco_gt(gt_dict_mot_cat, image_size=detection_dict['image_size'], cat_id_override=1, categories=labels_detection)
        # Save detection and gt as coco format for evaluation
        gt_det_coco_file = os.path.join(args.output, 'gt_det_coco_format.json')
        save_json_file(gt_dict_coco_det, gt_det_coco_file)
        det_coco_file = save_coco_format(
            detection_dict['detections'],
            os.path.join(args.output, 'det_coco_format'),
            image_size=detection_dict['image_size'],
            labels=labels_detection,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(gt_det_coco_file, det_coco_file, save_path=os.path.join(args.output, 'detection_eval_results_%s.json' % (os.path.basename(det_model).split('.')[0])), log=log)
        print_and_log('Detection evaluation results: %s' % (str(eval_results)), log=log)

    # Tracking
    if check_gui_stop(log=log): return 0
    tracking_dict = track(
        my_video,
        detection_dict,
        args.output,
        device=args.device,
        tracking_size=args.tracking_size,
        score=args.det_score,
        log=log,
        tracker_type=args.tracker_type,
    )

    if args.eval_tracking:
        # Save tracking results in MOT format for evaluation
        track_mot_file = save_mot_format(
            tracking_dict['detections'],
            os.path.join(args.output, 'track_mot_format'),
            image_size=detection_dict['image_size'],
            labels=labels_detection,
            boundaries=boundaries
        )
        # Save gt in MOT format for evaluation
        gt_track_mot_folder = os.path.join(args.output, 'gt_track_mot_format')
        save_mot_format(
            gt_dict_mot_cat,
            gt_track_mot_folder,
            labels=labels_detection,
            boundaries=boundaries,
            cat_id_override=1
        )
        # Evaluate tracking results
        eval_results = evaluate_tracking(gt_track_mot_folder, track_mot_file, save_path=os.path.join(args.output, 'tracking_eval_results_%s.txt' % (args.tracker_type if args.tracker_type else 'default')), log=log)
        print_and_log('Tracking evaluation results:\n%s' % (str(eval_results)), log=log)
    classes = ['NoID']

    # Classification
    if check_gui_stop(log=log): return 0
    classification_dict = classify(
        tracking_dict,
        my_video,
        args.output,
        log=log
    )
    if check_gui_stop(log=log): return 0

    if args.eval_classification:
        # Evaluate classification results using COCO metrics
        gt_dict_coco_class = mot_gt_to_coco_gt(gt_dict_mot_cat, image_size=classification_dict['image_size'], categories=classes)
        gt_class_coco_file = save_coco_format(gt_dict_coco_class['detections'], os.path.join(args.output, 'gt_class_coco_format'), labels=classes)
        class_coco_file = save_coco_format(
            classification_dict['detections'],
            os.path.join(args.output, 'class_coco_format'),
            image_size=classification_dict['image_size'],
            labels=classes,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(gt_class_coco_file, class_coco_file, log=log)
        save_json_file(eval_results, os.path.join(args.output, 'classification_eval_results.json'))
        print_and_log('Classification evaluation results: %s' % (str(eval_results)), log=log)

    # Save MOT format
    save_mot_format(
        classification_dict['detections'],
        os.path.join(args.output, 'mot'),
        image_size=classification_dict['image_size'],
        labels=classes
    )

    # Save COCO format
    save_coco_format(
        classification_dict['detections'],
        os.path.join(args.output, 'coco'),
        image_size=classification_dict['image_size'],
        labels=classes
    )

    # Create the video
    if args.video_demo:
        my_video.reset_video()
        my_video.plot_annotations(
            classification_dict['detections'],
            os.path.join(args.output, 'video_demo_%s.mp4' % (args.tracker_type if args.tracker_type else 'default')),
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

    # Check tracker type
    assert args.tracker_type in [None, 'bytetrack', 'deepsort', 'botsort'], 'Tracker type must be one of None, bytetrack, deepsort or botsort.'
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
        default=60,
        help=helptext_tracking_size
    )
    parser.add_argument(
        '-T', '--tracker_type',
        type=str,
        default=None,
        help=helptext_tracker_type
    )
    parser.add_argument(
        '-e', '--eval_detection',
        action='store_true',
        help=helptext_eval_detection
    )
    parser.add_argument(
        '-E', '--eval_tracking',
        action='store_true',
        help=helptext_eval_tracking
    )
    parser.add_argument(
        '-C', '--eval_classification',
        action='store_true',
        help=helptext_eval_classification
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
