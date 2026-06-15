import sys
import os
import cv2
import numpy as np
import time
import datetime
import copy
import torch
torch.cuda.init()
from argparse import ArgumentParser
import pdb
from scipy.ndimage import gaussian_filter
from collections import defaultdict

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
from .bt_utils.eval_utils import evaluate_detection, extract_boundaries, evaluate_tracking, mot_gt_to_coco_gt, save_mot_format,\
    save_coco_format, load_mot_format, coco_to_perso_format, perso_format_to_trackid_format, solve_id_conflicts
from .bt_utils.sam3_utils import process_video_with_sam, compute_mask_iou
from .bt_utils.classifier import load_model_and_transform, extract_feature, build_feature_dict, build_image_paths_dict,\
    get_class_scores, resolve_class_assignments

# Help variables
from .help import *

def load_as_detection_dict(var, image_size=None, log=None):
    # Load detection results if not already loaded
    if isinstance(var, str):
        print_and_log('Loading detection_dict from %s' % (var), log=log)
        var = load_json_file(var)
    if isinstance(var, list):
        var = {'detections': coco_to_perso_format(var, image_size=image_size)}
    return var

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

def process_detections(detections, last_tracks, track_id, image_size, score, iou_threshold=0.3, log=None):
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
        log: logger, the logger to print the information (default None)
    Returns:
        int, the updated track ID after assignment
    '''
    to_delete = []
    matches = []
    for idx, detection in enumerate(detections):
        # Remove detections with low score
        if detection.get('score', detection.get('conf', 0)) < score:
            print_and_log('Detection with low score found (frame %d): %f' % (idx, detection['conf']), log=log)
            to_delete.append(idx)
            continue
        # Match track ID based on IoU with last tracks
        best_iou = iou_threshold
        for track in last_tracks:
            if 'segmentation' in track and 'segmentation' in detection:
                iou = compute_mask_iou(track, detection)
            else:
                iou = compute_iou(detection['bbox'], track['bbox'], image_size)
            if iou > best_iou:
                matches.append({'det_idx':idx, 'track_id': track['track_id'], 'iou': iou})
                best_iou = iou
        # Remove existing track_id if exists to avoid confusion with the new assigned track_id
        if 'track_id' in detection:
            del detection['track_id']

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

        # Megadetector + coco format compatibility
        detection['category_id'] = int(detection.get('category', detection.get('category_id', 1)))
        detection['score'] = detection.get('score', detection.get('conf', 0))
        detection['visibility'] = get_value_with_precision(
            1-max([match['iou'] for match in matches if match['det_idx'] == idx and match['track_id'] != detection['track_id']], default=0)
        )
        # Remove the normalized keys
        if 'category' in detection: del detection['category']
        if 'conf' in detection: del detection['conf']

    # Remove detections with low score
    for idx in reversed(to_delete):
        del detections[idx]

    return track_id


def detect(my_video, output_file, device='cpu', tracking_size=60, score=0.5, det_model='md_v5b.0.0.pt', text_prompt="an animal", log=None, display_fct=None):
    '''
    Detect the Baboons in the video using Megadetector and track them.

    Args:
        my_video: VideoFrameIterator, the video to process
        output_file: str, the path to save the output file
        device: str, the device to use for the detection
        tracking_size: int, the size of the tracking buffer in number of frames (default 60)
        score: float, the minimum detection score (default 0.5)
        display_fct: function, the function to display the results in real-time (default None)
        det_model: str, the path to the detection model (default 'md_v5b.0.0.pt')
        text_prompt: str, the text prompt to use for SAM 3 tracking (default "an animal")
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the detection and tracking results
    '''
    # Initialization
    ## Check if the output file already exists
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Skipping detection and tracking.' % (output_file), log=log)
        return output_file

    ## Variables
    track_id = 0
    det_results = []
    tracking_buffer = []
    last_tracks = []
    start_time = time.time()

    ## SAM3
    if "sam3" in det_model:
        return process_video_with_sam(my_video, output_file, text_prompt=text_prompt, chunk_size=200, overlap=5, tmp_dir=".tmp", clean_up=False, det_only='det' in det_model, log=log)
    
    ## Megadetector
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
                '(%ds left)' % (elapsed_time/idx*(len(my_video)-idx)) if idx else ''
            )
        )
        
        ## Detection on RGB ordered image (Could be improved by running the detection on a batch of frames instead of one by one) 
        det_result = model.generate_detections_one_image(frame, detection_threshold=score)['detections'] # image_id=idx

        ## Assign track_ids
        track_id = process_detections(det_result, last_tracks, track_id, image_size, score, iou_threshold=0.3, log=log)

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

def track(my_video, detection_dict, output_file, device='cpu', tracking_size=60, score_th=0.5, tracker_type=None, image_size=None, log=None):
    '''
    Track the Baboons in the video using the detection results.

    Args:
        my_video: VideoFrameIterator, the video to process
        detection_dict: dict, the detection results
        output_file: str, the path to save the output file
        device: str, the device to use for the tracking
        tracking_size: int, the size of the tracking buffer in number of frames (default 30)
        score_th: float, score threshold for detections to be considered for tracking (default 0.5)
        tracker_type: str, the type of tracker to use (default None)
        image_size: tuple, the size of the image (width, height)
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the tracking results
    '''
    # Initialization
    ## Check if the output file already exists
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Skipping tracking. Loading existing file.' % (output_file), log=log)
        return output_file
    
    ## Check if detection_dict is filepath or dict
    detection_dict = load_as_detection_dict(detection_dict, image_size=image_size, log=log)

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
    elif tracker_type == "IoU":
        print_and_log('Tracking with IoU', log=log)
        track_id = 0
        tracking_buffer = []
        last_tracks = []
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
                '(%ds left)' % (elapsed_time/idx*(len(my_video)-idx)) if idx else ''
            )
        )

        ## Set image size once from the first frame
        if idx == 0:
            image_size = detection_dict['image_size'] if 'image_size' in detection_dict else frame.shape[:2][::-1]

        ## Update tracker with the detection results
        if tracker_type in ["bytetrack", "botsort"]:
            # Bboxes in (x1, y1, x2, y2) format pixel coordinates and scores
            bboxes = []
            for det in det_result:
                x, y, w, h = det['bbox']
                x1 = int(x * image_size[0])
                y1 = int(y * image_size[1])
                x2 = int((x + w) * image_size[0])
                y2 = int((y + h) * image_size[1])
                bboxes.append([x1, y1, x2, y2])
            bboxes = np.array(bboxes)
            # Use "score" or "det_score" according to availability
            scores = np.array([det['score'] for det in det_result])
            if tracker_type == "bytetrack":
                _current_tracks = tracker.update(bboxes, scores)
            elif tracker_type == "botsort":
                # Expecting bboxes in (x1, y1, x2, y2) format pixel coordinates, scores and classid (same classid for all detections here)
                bboxes_scores = np.hstack((bboxes, scores[:, np.newaxis], np.ones((len(scores), 1)))) if len(scores) > 0 else np.empty((0, 6))
                _current_tracks = tracker.update(bboxes_scores, frame)
            # Convert back to (x, y, w, h) format and normalized coordinates
            current_tracks = []
            for track in _current_tracks:
                x1, y1, w, h = track.tlwh
                tmp = {}
                tmp['bbox'] = [float(x1 / image_size[0]), float(y1 / image_size[1]), float(w / image_size[0]), float(h / image_size[1])]
                tmp['track_id'] = track.track_id
                tmp['score'] = float(track.score)
                n_tracks = max(n_tracks, track.track_id)
                current_tracks.append(tmp)
            all_tracks.append(current_tracks)
        elif tracker_type == "deepsort":
            current_tracks, max_track_id = update_tracker(tracker, frame, feat_model, det_result)
            n_tracks = max(n_tracks, max_track_id)
            all_tracks.append(current_tracks)
        elif tracker_type == "IoU":
            ## Assign track_ids
            track_id = process_detections(det_result, last_tracks, track_id, image_size, score_th, iou_threshold=0.3, log=log)
            n_tracks = track_id - 1
            ## Update Tracking
            if len(tracking_buffer) > tracking_size:
                tracking_buffer.pop(0)
            tracking_buffer.append(det_result)
            last_tracks = get_last_tracks(tracking_buffer)

            ## Save and display results
            all_tracks.append(det_result)
    progress_bar(len(my_video), len(my_video), 'Tracking done in %ds with %d tracks' % (time.time() - start_time, n_tracks), log=log, completed=True)

    # Saving
    output_results = {'detections': all_tracks, 'format': 'xywh', 'image_size': image_size, 'n_tracks': n_tracks, 'tracker_type': tracker_type}
    if 'detection_classes' in detection_dict:
        output_results['detection_classes'] = detection_dict['detection_classes']
    save_json_file(output_results, output_file)
    return output_results

def classify(detection_dict, my_video, output_file, class_database=None, class_threshold=0.5, image_size=None, device='cpu', log=None):
    '''
    Classify the tracks of the detected Baboons using a pre-trained classifier and a dictionary with extracted features from the tracks.

    Args:
        detection_dict: dict, the detection and tracking results
        my_video: VideoFrameIterator, the video to process
        output_file: str, the path to save the output file
        class_database: str, the path to the classification dictionary
        class_threshold: float, the threshold for class assignment
        image_size: tuple, the size of the image (width, height)
        device: str, the device to use for feature extraction (default 'cpu')
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the classification results
    '''
    # Initialization
    start_time = time.time()
    ## Check if the output file already exists
    if os.path.exists(output_file):
        print_and_log('Output file %s already exists. Loading existing file.' % (output_file), log=log)
        return load_json_file(output_file)
    ## Classification names
    if class_database:
        classes = sorted(os.listdir(class_database))
        img_path_dict = build_image_paths_dict(class_database)  # Check if the classification dictionary is well formed
    else:
        classes = []
        img_path_dict = {}
    classes.append('NoID')
    n_tracks = 0
    model, transform = load_model_and_transform(device=device)
    feature_database = build_feature_dict(model, transform, img_path_dict)

    ## Check if detection_dict is filepath or dict
    detection_dict = load_as_detection_dict(detection_dict, image_size=image_size, log=log)
    class_dict = copy.deepcopy(detection_dict)

    ## Check if Steps 1 and 2 are already done
    score_file = output_file.replace('.json', '_with_scores.json')
    if os.path.exists(score_file):
        print_and_log('Score file %s already exists. Loading existing file with scores.' % (score_file), log=log)
        track_class_dict = load_json_file(score_file)['track_class_dict']
    else:
        ## Step 1: Sort dict per track_id
        track_dict = perso_format_to_trackid_format(class_dict['detections'])

        ## Step 2: For each track, extract the features per image and get the class scores
        track_class_dict = defaultdict(list)
        for track_id, track_dets in track_dict.items():
            elapsed_time = time.time() - start_time
            progress_bar(
                track_id,
                len(track_dict),
                'Classifying tracks.%s' % ('(%ds left)' % (elapsed_time/track_id*(len(track_dict)-track_id)) if len(track_class_dict) > 0 else '')
            )
            features = []
            idxs = []
            for det in track_dets:
                idx = det['image_id']-1  # image_id starts at 1 in coco format
                idxs.append(idx)
                img = my_video.get_frame_at_idx(idx)
                x, y, w, h = det['bbox']
                img_cropped = img[int(y*img.shape[0]):int((y+h)*img.shape[0]), int(x*img.shape[1]):int((x+w)*img.shape[1])]
                feature = extract_feature(model, transform, img_cropped)
                features.append(feature)
            track_class_dict[track_id] = {
                'scores': get_class_scores(features, feature_database),
                'idxs': idxs
            }
        progress_bar(len(track_dict), len(track_dict), 'Classifying tracks done in %ds. Resolving assignments...' % (time.time() - start_time), log=log, completed=True)
        # Save the intermediate results with the scores before resolving the final class assignments to avoid losing information in case of crash and for debugging purposes
        save_json_file({'track_class_dict': track_class_dict, 'class_database': classes}, score_file)

    ## Step 3: Final decision on the class of each track based on the scores, a threshold and overlapping tracks using while. If no score is above the threshold, assign "NoID" class.
    final_assignments = resolve_class_assignments(track_class_dict, class_threshold=class_threshold)

    ## Step 4: Save the results in in class_dict
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for dets_per_frame in class_dict['detections']:
        for det in dets_per_frame:
            track_id = det['track_id']
            assigned_class, assigned_score = final_assignments.get(track_id,("NoID", 0.0))
            n_tracks = max(n_tracks, track_id)
            # Save initial detection 
            det['det_id'] = det.get('category_id', 1)
            det['det_score'] = det.get('score')
            # Reassign the class and score based on the classification results
            det['category_id'] = class_to_idx[assigned_class]
            det['score'] = assigned_score
    # Saving
    # class_dict['detection_classes'] = [f"Track {i}" for i in range(1, n_tracks + 1)]
    class_dict['classification_classes'] = classes
    save_json_file(class_dict, output_file)
    print_and_log('Classification done in %ds for %d tracks. Results saved in %s' % (time.time() - start_time, n_tracks, output_file), log=log)

    return class_dict

def false_check(log=None):
    '''
    False check function.

    Args:
        log: logger, the logger to print the information (default None)
    
    Returns:
        bool, False, to indicate that the process should not be stopped.
    '''
    return False

def main(args, check_stop=false_check, gt_file_class_mot=None, log=None):
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
    # Path checks and output folder creation
    if not os.path.exists(args.input_video):
        print_and_log('Error: input %s must be a file or a folder' % (args.input_video), log=log, to_print=False)
        raise ValueError('No input provided.')
    os.makedirs(args.output, exist_ok=True)
    
    # Chrono
    start_time = time.time()

    # Video object initialization
    my_video = VideoFrameIterator(args.input_video, log=log)
    image_size = my_video.get_image_size()
    print_and_log('Video %s opened with resolution %s and %d frames.' % (args.input_video, str(image_size), len(my_video)), log=log)
    # my_video.check_video()

    # Load ground truth if evaluation is enabled
    if (args.eval_detection or args.eval_tracking or args.eval_classification) and gt_file_class_mot:
        # With class ID being the track id but also the det ID
        boundaries = extract_boundaries(gt_file_class_mot)
        gt_dict_mot_cat, gt_labels = load_mot_format(gt_file_class_mot, boundaries=boundaries)

    # Detection
    if check_stop(log=log): return 0
    det_model_str = '%s%s' % (os.path.basename(args.det_model).split('.')[0], ("_" + args.text_prompt.replace(" ", "_")) if 'sam3' in args.det_model else "")
    detection_dict = detect(
        my_video,
        os.path.join(args.output, 'det_%s.json') % det_model_str,
        device=args.device,
        score=args.det_score_th,
        tracking_size=args.tracking_size,
        det_model=args.det_model,
        text_prompt=args.text_prompt,
        log=log,
        display_fct=args.display_fct
    )

    if args.eval_detection and gt_file_class_mot:
        # Load detection results if not already loaded
        detection_dict = load_as_detection_dict(detection_dict, image_size=image_size, log=log)
        # Convert MOT GT to COCO format for evaluation
        labels_detection = detection_dict.get('detection_classes', [args.text_prompt] if 'sam3' in args.det_model else None)
        gt_dict_coco_det = mot_gt_to_coco_gt(gt_dict_mot_cat, image_size=image_size, cat_id_override=1, categories=labels_detection)
        # Save detection and gt as coco format for evaluation
        gt_det_coco_file = os.path.join(args.output, 'gt_det_coco_format.json')
        save_json_file(gt_dict_coco_det, gt_det_coco_file)
        det_coco_file = save_coco_format(
            detection_dict['detections'],
            os.path.join(args.output, 'det_coco_format_%s' % det_model_str),
            image_size=image_size,
            labels=labels_detection,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(
            gt_det_coco_file,
            det_coco_file,
            name='%s%s' % (det_model_str, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            save_path=os.path.join(args.output, 'det_eval.csv'),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
        )
        print_and_log('Detection evaluation results: %s' % (str(eval_results)), log=log)

    # Tracking
    if check_stop(log=log): return 0
    det_tracker_str = '%s_%s' % (det_model_str, args.tracker_type)
    tracking_dict = track(
        my_video,
        detection_dict,
        os.path.join(args.output, 'track_%s.json' % det_tracker_str),
        device=args.device,
        tracking_size=args.tracking_size,
        score=args.det_score_th,
        tracker_type=None if args.tracker_type == 'IoU' and 'sam3' not in args.det_model else args.tracker_type,
        image_size=image_size,
        log=log
    )

    if args.eval_tracking and gt_file_class_mot:
        # Load tracking results if not already loaded
        tracking_dict = load_as_detection_dict(tracking_dict, image_size=image_size, log=log)
        # Save tracking results in MOT format for evaluation
        track_mot_file = save_mot_format(
            tracking_dict['detections'],
            os.path.join(args.output, 'track_mot_format'),
            image_size=image_size,
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
        eval_results = evaluate_tracking(
            gt_track_mot_folder,
            track_mot_file,
            name='%s%s' % (det_tracker_str, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            save_path=os.path.join(args.output, 'track_eval.csv'),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
        )
        print_and_log('Tracking evaluation results:\n%s' % (str(eval_results)), log=log)
    
    # Classification
    if check_stop(log=log): return 0
    class_dict = classify(
        tracking_dict,
        my_video,
        os.path.join(args.output, 'class_%s.json' % (det_tracker_str)),
        class_database=args.class_database,
        image_size=image_size,
        device=args.device,
        log=log
    )
    if check_stop(log=log): return 0
    classes = class_dict['classification_classes']
    
    if args.eval_classification and gt_file_class_mot:
        # Evaluate classification results using COCO metrics
        gt_dict_coco_class = mot_gt_to_coco_gt(gt_dict_mot_cat, image_size=image_size, categories=gt_labels)
        gt_class_coco_file = os.path.join(args.output, 'gt_class_coco_format.json')
        save_json_file(gt_dict_coco_class, gt_class_coco_file)
        # gt_class_coco_file = save_coco_format(gt_dict_coco_class['annotations'], os.path.join(args.output, 'gt_class_coco_format'), labels=gt_labels)
        uniform_class_list = solve_id_conflicts(class_dict['detections'], classes, gt_labels, default_label='NoID', log=log)
        class_coco_file = save_coco_format(
            uniform_class_list,
            os.path.join(args.output, 'class_coco_format'),
            image_size=image_size,
            labels=classes,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(
            gt_class_coco_file,
            class_coco_file,
            save_path=os.path.join(args.output, 'class_eval.csv'),
            name='%s%s' % (det_tracker_str, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
        )
        print_and_log('Classification evaluation results: %s' % (str(eval_results)), log=log)

    # Save MOT format
    save_mot_format(
        class_dict['detections'],
        os.path.join(args.output, 'mot'),
        image_size=image_size,
        labels=classes
    )

    # # Save COCO format
    # save_coco_format(
    #     class_dict['detections'],
    #     os.path.join(args.output, 'coco'),
    #     image_size=image_size,
    #     labels=classes
    # )

    # Create the video
    if args.video_demo:
        my_video.reset_video()
        my_video.plot_annotations(
            class_dict['detections'],
            os.path.join(args.output, 'video_demo_%s.mp4' % (det_tracker_str)),
            max_res=args.max_res,
            display_fct=args.display_fct,
            detection_classes=class_dict.get('detection_classes', [args.text_prompt] if 'sam3' in args.det_model else None),
            classification_classes=class_dict.get('classification_classes'),
            n_tracks=tracking_dict.get('n_tracks', None),
            del_imgs=args.del_imgs,
            log=log
        )
        if check_stop(log=log): return 0

    print_and_log("Processing of %s finished in %ds." % (args.input_video, time.time()-start_time), log=log)

def main_loop(args, log=None):
    '''
    Main loop to process the video. It allows to loop over the video and process it multiple times
    using different combination of parameters.

    Args:
        args: argparse.Namespace, the arguments
        log: logger, the logger to print the information
    '''
    det_models = ['MDv5a', 'MDv5b', 'sam3', 'sam3_det']
    det_models = ['sam3']  # for testing
    prompts = ['an animal', 'a baboon', 'a monkey', 'a primate', 'an ape']
    prompts = ['a baboon']  # for testing
    tracker_types = ['IoU', 'bytetrack', 'deepsort', 'botsort', 'sam3']
    tracker_types = ['sam3']  # for testing
    gt_files_name = ['frame-1639-2000-mot.zip', 'frame-1-546-mot.zip']
    for det_model in det_models:
        args.det_model = det_model
        for tracker_type in tracker_types:
            if tracker_type == 'sam3' and det_model != 'sam3':
                continue
            args.tracker_type = tracker_type
            for prompt in prompts if 'sam3' in det_model else ['']:
                args.text_prompt = prompt
                print_and_log('Running det %s%s and tracker %s' % (det_model, ' with prompt "%s"' % (prompt) if prompt else '', tracker_type), log=log)
                for gt_file_class_mot in [os.path.join(os.path.dirname(args.input_video), name) for name in gt_files_name]:
                    main(args, gt_file_class_mot=gt_file_class_mot, log=log)


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
    assert args.tracker_type in ['IoU', 'bytetrack', 'deepsort', 'botsort', 'sam3'], 'Tracker type must be one of IoU, bytetrack, deepsort or botsort.'

    # Check if tracker type is compatible with detection model
    if 'sam3' in args.tracker_type:
        assert 'sam3' in args.det_model, 'Tracker type %s is only compatible with sam3 detection model.' % (args.tracker_type)
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
        '-s', '--det_score_th',
        default=0.5,
        type=float,
        help=helptext_det_score_th
    )
    parser.add_argument(
        '-c', '--del-imgs',
        action='store_true',
        help=helptext_del_imgs
    )
    parser.add_argument(
        '-d', '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help=helptext_device
    )
    parser.add_argument(
        '-D', '--det_model',
        default='md_v5b.0.0.pt',
        type=str,
        help=helptext_det_model
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
        default='IoU',
        help=helptext_tracker_type
    )
    parser.add_argument(
        '-p', '--text_prompt',
        type=str,
        default="a baboon",
        help=helptext_text_prompt
    )
    parser.add_argument(
        '-P', '--class_database',
        default=os.path.join('/shared', 'group_dict'),
        type=str,
        help=helptext_class_database
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
    parser.add_argument(
        '-l', '--loop',
        action='store_true',
        help=helptext_loop
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

    main_funct = main_loop if args.loop else main

    # Send arguments to main
    if args.gui:
        # Use GUI
        from .gui import run_with_gui
        run_with_gui(args, main_funct, check_args_fct=infer_args_name)
    else:
        os.makedirs(os.path.join(args.output, 'logs'), exist_ok=True)
        log = setup_logger(log_file=os.path.join(args.output, 'logs', '%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))))
        print_and_log('Starting BaboonTrack without GUI with arguments: %s' % (args), log=log)
        # Check if input is video or has frames. If not, loop over the folder and process each video or set of frames separately.
        if os.path.isfile(args.input_video) or (os.path.isdir(args.input_video) and any(os.path.isfile(os.path.join(args.input_video, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(args.input_video))):
            main_funct(args, log=log)
        else:
            input_list = sorted([os.path.join(args.input_video, f) for f in os.listdir(args.input_video)])
            main_output = copy.deepcopy(args.output)
            for input_path in input_list:
                args.input_video = input_path
                args.output = os.path.join(main_output, os.path.basename(input_path).split('.')[0])
                main_funct(args, log=log)
        close_log(log)
    
    # Finish
    print('BaboonTrack finished.')
    return 1

if __name__ == '__main__':
    # Run BaboonTrack
    run()

    # Exit
    sys.exit(0)
