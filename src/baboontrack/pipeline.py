import pdb
import sys
import os
import cv2
import numpy as np
import time
import datetime
import copy
import torch
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Megadetector
from megadetector.detection import run_detector

# Print Versions
print('Python version: %s' % (sys.version))
print('OpenCV version: %s' % (cv2.__version__))
print('PyTorch version: %s' % (torch.__version__))

# Utility functions
from .args import check_args, infer_args_name
from .bt_utils.io_utils import print_and_log, setup_logger, close_log, progress_bar, get_value_with_precision, find_file_with_ending
from .bt_utils.json_utils import save_json_file, load_json_file
from .bt_utils.img_utils import VideoFrameIterator, apply_roi_factor
from .bt_utils.plt_utils import save_img_with_bbox
from .bt_utils.tracking import ReIDModel, init_tracker, update_tracker
from .bt_utils.eval_utils import evaluate_detection, extract_boundaries, evaluate_tracking, mot_gt_to_coco_gt, save_mot_format,\
    save_coco_format, load_mot_format, coco_to_perso_format, perso_format_to_trackid_format, solve_id_conflicts, merge_eval_coco, \
    merge_mot_formats
from .bt_utils.sam3_utils import process_video_with_sam, compute_mask_iou
from .bt_utils.classifier import MyClassifier, build_image_paths_dict, resolve_class_assignments

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


def detect(my_video, output_file, device='cpu', tracking_size=60, score=0.5, det_model='md_v5b.0.0.pt', text_prompt="an animal", chunk_size=200, overlap=5, log=None, display_fct=None):
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
    if not my_video.checked:
        my_video.check_video()

    ## Variables
    track_id = 1
    det_results = []
    tracking_buffer = []
    last_tracks = []
    start_time = time.time()

    ## SAM3
    if "sam3" in det_model:
        return process_video_with_sam(my_video, output_file, text_prompt=text_prompt, chunk_size=chunk_size, overlap=overlap, tmp_dir=".tmp", clean_up=False, det_only='det' in det_model, log=log)
    
    ## Megadetector
    model = run_detector.load_detector(
        det_model,
        detector_options={'device':device}
    ) # 1452MB on gpu
    det_classes = ["animal","person","vehicle"]

    # Loop over the video frames
    print_and_log('Processing video %s' % (my_video.path), log=log)
    start_loop = time.time()
    for idx, frame in enumerate(my_video):
        if idx == 0:
            image_size = frame.shape[:2][::-1]
            print_and_log('Video resolution: %s' % (str(image_size)), log=log)           
        elapsed_time = time.time() - start_loop
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
    progress_bar(len(my_video), len(my_video), 'Detection and Tracking done in %ds with %d tracks' % (time.time() - start_time, track_id-1), log=log, completed=True)
        
    # Saving
    track_ids = [i for i in range(1, track_id)]
    output_results = {'detections': det_results, 'detection_classes': det_classes, 'format': 'xywh', 'image_size': image_size, 'track_ids': track_ids}
    save_json_file(output_results, output_file)
    save_json_file(output_results, output_file.replace('.json', '.pretty.json'), pretty=True)
    return output_results

def track(my_video, detection_dict, output_file, device='cpu', tracking_size=60, score_th=0.5, tracker_type=None, image_size=None, default_cat_id=1, log=None):
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
        default_class_id: int, the default class ID to assign to detections (default 1)
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
    if not my_video.checked:
        my_video.check_video()
    n_tracks = 0
    all_tracks = []
    start_loop = time.time()
    for idx, (frame, det_result) in enumerate(zip(my_video, detection_dict['detections'])):
        ## Progress bar with estimated time remaining
        elapsed_time = time.time() - start_loop
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
                tmp['category_id'] = default_cat_id
                n_tracks = max(n_tracks, track.track_id)
                current_tracks.append(tmp)
            all_tracks.append(current_tracks)
        elif tracker_type == "deepsort":
            current_tracks, max_track_id = update_tracker(tracker, frame, feat_model, det_result, default_cat_id=default_cat_id)
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
    track_ids = [i for i in range(1, n_tracks+1)]
    output_results = {'detections': all_tracks, 'format': 'xywh', 'image_size': image_size, 'track_ids': track_ids, 'tracker_type': tracker_type}
    if 'detection_classes' in detection_dict:
        output_results['detection_classes'] = detection_dict['detection_classes']
    save_json_file(output_results, output_file)
    save_json_file(output_results, output_file.replace('.json', '.pretty.json'), pretty=True)
    return output_results

def extract_and_store_feature(classifier, img, roi_file, roi_dir, track_id, frame_idx, features):
    """
    Extract a feature and store it in the feature dictionary.

    Args:
        classifier: MyClassifier, the classifier to use for feature extraction
        img: np.ndarray, the image to extract the feature from
        roi_file: str, the path to the region of interest file
        roi_dir: str, the directory to save the region of interest with bounding box
        track_id: int, the track ID
        frame_idx: int, the frame index
        features: dict, the dictionary to store the extracted features

    Returns:
        np.ndarray: The extracted bounding box coordinates, or None if extraction failed.
    """
    feature, bbox = classifier.extract_feature(img if img is not None else roi_file)
    if feature is None:
        return None
    if bbox is not None:
        crop_path = os.path.join(roi_dir, f"track_{track_id}_with_bbox", f"{frame_idx}.jpg")
        if img is not None:
            save_img_with_bbox(img, bbox, crop_path)
    else:
        crop_path = roi_file
    features[crop_path] = feature
    return bbox

def classify(detection_dict, my_video, output_file, class_database='', sim_th=0.5, image_size=None, device='cpu',
             class_det=None, class_det_thr=0.5, class_nms_thr=0.4, feat_avg=None, nca=None, epochs=100, lr=1e-4, roi_factor=1.0,
             roi_det=1.0, avg_score=False, noid_str='NoID', source_roi='', joint_factor=0, log=None):
    '''
    Classify the tracks of the detected Baboons using a pre-trained classifier and a dictionary with extracted features from the tracks.

    Args:
        detection_dict: dict, the detection and tracking results
        my_video: VideoFrameIterator, the video to process
        output_file: str, the path to save the output file
        class_database: str, the path to the classification dictionary
        sim_th: float, similarity threshold for class assignment (default 0.5)
        image_size: tuple, the size of the image (width, height)
        device: str, the device to use for feature extraction (default 'cpu')
        class_det: str, the type of detector to use for classification (default None)
        class_det_thr: float, the threshold for detection (default 0.5)
        class_nms_thr: float, the threshold for non-maximum suppression (default 0.4)
        feat_avg: np.ndarray, the average features for each class (default None)
        nca: object, the Neighborhood Component Analysis object (default None)
        epochs: int, the number of epochs for training (default 100)
        lr: float, the learning rate for training (default 1e-4)
        roi_factor: float, the factor to scale the region of interest for feature extraction (default 1.0)
        roi_det: float, the factor to scale the region of interest for detection (default 1.0)
        noid_str: str, the string for the "NoID" class (default 'NoID')
        source_roi: str, the source of the region of interest (default '') - allow loading faster
        joint_factor: float, the factor to combine the features from two classifiers (default 0)
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

    ## Check if detection_dict is filepath or dict
    detection_dict = load_as_detection_dict(detection_dict, image_size=image_size, log=log)
    class_dict = copy.deepcopy(detection_dict)
    classes = [noid_str]

    ## Classification routine
    if class_database:
        classes += sorted(os.listdir(class_database))
        img_path_dict = build_image_paths_dict(class_database)  # Check if the classification dictionary is well formed

        ## Step 1: Sort dict per track_id
        track_dict = perso_format_to_trackid_format(class_dict['detections'])

        ## Step 2: For each track, extract the features per image and get the class scores
        if source_roi:
            roi_path = os.path.join(os.path.dirname(output_file), 'rois', source_roi) 
            os.makedirs(roi_path, exist_ok=True)
        # Remove sim_th from the filename to allow reusing the same track_class_dict for different sim_th values
        track_class_dict_path = os.path.join(os.path.dirname(output_file), 'track_class', os.path.basename(output_file).replace('_joint-%g' % joint_factor, '_joint').replace('_simth-%g' % sim_th, ''))
        os.makedirs(os.path.dirname(track_class_dict_path), exist_ok=True)
        if os.path.exists(track_class_dict_path):
            print_and_log('Loading track_class_dict from %s' % (track_class_dict_path), log=log)
            track_class_dict = load_json_file(track_class_dict_path)
            track_class_dict = {int(k): v for k, v in track_class_dict.items()}  # Convert keys to int
        else:
            # Build the database of features for the classifier
            if joint_factor:
                print_and_log('Joint classification is enabled. All tracks will be classified using two preset_classifiers.', log=log)
                # Full image
                my_classifier = MyClassifier(device=device, detector_type=None, det_thr=0, nms_thr=0, feat_avg=False, nca=True, epochs=200,
                                                lr=0.0001, roi_det=1, avg_score=avg_score, name_database=os.path.basename(class_database), log=log)
                # Primate face crop
                my_classifier2 = MyClassifier(device=device, detector_type='primateface', det_thr=0.5, nms_thr=0.4, feat_avg=False, nca=True,
                                                epochs=200, lr=0.0001, roi_det=2.5, avg_score=avg_score, name_database=os.path.basename(class_database),
                                                log=log)
            else:
                my_classifier = MyClassifier(device=device, detector_type=class_det, det_thr=class_det_thr, nms_thr=class_nms_thr,
                                            feat_avg=feat_avg, nca=nca, epochs=epochs, lr=lr, roi_det=roi_det, avg_score=avg_score,
                                            name_database=os.path.basename(class_database), log=log)
            print_and_log('Building the database of features for the classifier from %s' % (class_database), log=log)
            my_classifier.build_database(img_path_dict)
            if joint_factor:
                print_and_log('Building the database of features for the second classifier (primateface) from %s' % (class_database), log=log)
                my_classifier2.build_database(img_path_dict)
            track_class_dict = defaultdict(list)
            start_loop = time.time()
            for idx, (track_id, track_dets) in enumerate(track_dict.items()):
                elapsed_time = time.time() - start_loop
                progress_bar(
                    idx,
                    len(track_dict),
                    'Classifying tracks.%s' % ('(%ds left)' % (elapsed_time/idx*(len(track_dict)-idx)) if idx > 0 else '')
                )
                track_features_path = os.path.join(
                    os.path.dirname(output_file),
                    'features',
                    os.path.basename(output_file).replace('_joint-%g' % joint_factor, '_joint').replace('_avg_score', '').replace('_simth-%g' % sim_th, '').replace('.json', ''),
                    'track_%d.pt' % track_id
                )
                if os.path.exists(track_features_path):
                    print_and_log('Loading features for track %d from %s' % (track_id, track_features_path), log=log)
                    track_features = torch.load(track_features_path, map_location=device)
                    features = track_features['features']
                    if joint_factor:
                        features2 = track_features['features2']
                    idxs = track_features['idxs']
                    extra_bboxs = track_features['extra_bboxes']
                else:
                    features = {}
                    if joint_factor:
                        features2 = {}
                    idxs = []
                    extra_bboxs = {}
                    ## Set my_video to bgr extraction for classification
                    if not my_video.checked:
                        my_video.check_video()
                    my_video.bgr = True
                    ### If source_roi is provided and the roi files already exist, load them instead of extracting them again
                    if source_roi and os.path.exists(os.path.join(roi_path, 'track_%d' % track_id)) and len(os.listdir(os.path.join(roi_path, 'track_%d' % track_id))) == len(track_dets):
                        for det in track_dets:
                            frame_idx = det['image_id']-1  # image_id starts at 1 in coco format
                            idxs.append(frame_idx)
                            roi_file = os.path.join(roi_path, 'track_%d' % track_id, '%d.jpg' % frame_idx)
                            extra_bbox = extract_and_store_feature(my_classifier, None, roi_file, roi_path, track_id, frame_idx, features)
                            if joint_factor:
                                extra_bbox2 = extract_and_store_feature(my_classifier2, None, roi_file, roi_path, track_id, frame_idx, features2)
                            # Save extra bbox in det for visualization in video
                            if extra_bbox is not None or (joint_factor and extra_bbox2 is not None):
                                extra_bboxs[frame_idx] = extra_bbox if extra_bbox is not None else extra_bbox2
                    else:
                        for det in track_dets:
                            frame_idx = det['image_id']-1  # image_id starts at 1 in coco format
                            idxs.append(frame_idx)
                            img = my_video.get_frame_at_idx(frame_idx)
                            x, y, w, h = apply_roi_factor(det['bbox'], roi_factor)
                            img_cropped = img[max(0, int(y*img.shape[0])):min(int((y+h)*img.shape[0]), img.shape[0]), max(0, int(x*img.shape[1])):min(int((x+w)*img.shape[1]), img.shape[1])]
                            if source_roi:
                                roi_file = os.path.join(roi_path, 'track_%d' % track_id, '%d.jpg' % frame_idx)
                                os.makedirs(os.path.dirname(roi_file), exist_ok=True)
                                cv2.imwrite(roi_file, img_cropped)
                            else:
                                roi_file = 'track_%d_frame_%d' % (track_id, frame_idx)
                            extra_bbox = extract_and_store_feature(my_classifier, img_cropped, roi_file, roi_path, track_id, frame_idx, features)
                            if joint_factor:
                                extra_bbox2 = extract_and_store_feature(my_classifier2, img_cropped, roi_file, roi_path, track_id, frame_idx, features2)
                            # Save extra bbox in det for visualization in video
                            if extra_bbox is not None or (joint_factor and extra_bbox2 is not None):
                                extra_bboxs[frame_idx] = extra_bbox if extra_bbox is not None else extra_bbox2
                    # Save the features, idxs and extra bboxes for this track in a dict file
                    track_features = {'features': {key: f.cpu() for key, f in features.items()}, 'idxs': idxs, 'extra_bboxes': extra_bboxs}
                    if joint_factor:
                        track_features['features2'] = {key: f.cpu() for key, f in features2.items()}
                    os.makedirs(os.path.dirname(track_features_path), exist_ok=True)
                    torch.save(track_features, track_features_path)
                scores, paths_ref, paths_crop = my_classifier.get_class_scores(features)
                track_class_dict[track_id] = {
                    'scores': scores,
                    'paths_ref': paths_ref,
                    'paths_crop': paths_crop,
                    'idxs': idxs,
                    'extra_bboxes': extra_bboxs
                }
                if joint_factor:
                    scores2, paths_ref2, paths_crop2 = my_classifier2.get_class_scores(features2)
                    track_class_dict[track_id]['scores2'] = scores2
                    track_class_dict[track_id]['paths_ref2'] = paths_ref2
                    track_class_dict[track_id]['paths_crop2'] = paths_crop2
            progress_bar(len(track_dict), len(track_dict), 'Classifying tracks done in %ds. Resolving assignments...' % (time.time() - start_loop), log=log, completed=True)
            # Save the track_class_dict for future use
            save_json_file(track_class_dict, track_class_dict_path)
        ## Step 3: Final decision on the class of each track based on the scores, a threshold and overlapping tracks.
        if joint_factor:
            print_and_log('Resolving class assignments with joint classification.', log=log)
            # Combine the scores from both classifiers by averaging them
            for track_id, track_info in track_class_dict.items():
                scores1 = track_info.get('scores', {})
                scores2 = track_info.get('scores2', {})
                paths_ref1 = track_info.get('paths_ref', {})
                paths_crop1 = track_info.get('paths_crop', {})
                paths_ref2 = track_info.get('paths_ref2', {})
                paths_crop2 = track_info.get('paths_crop2', {})
                scores = {}
                paths_ref = {}
                paths_crop = {}
                for cls_name in set(scores1.keys()).union(set(scores2.keys())):
                    scores[cls_name] = (1-joint_factor)*scores1.get(cls_name, 0.0) + joint_factor*scores2.get(cls_name, 0.0)
                    # Prefer paths from the classifier with more weight
                    paths_ref[cls_name] = paths_ref1.get(cls_name) if joint_factor < 0.5 else paths_ref2.get(cls_name)
                    paths_crop[cls_name] = paths_crop1.get(cls_name) if joint_factor < 0.5 else paths_crop2.get(cls_name)
                track_class_dict[track_id]['scores'] = scores
                track_class_dict[track_id]['paths_ref'] = paths_ref
                track_class_dict[track_id]['paths_crop'] = paths_crop
        final_assignments = resolve_class_assignments(track_class_dict, sim_th=sim_th)
    else:
        print_and_log(f'No classification dictionary provided. Assigning {noid_str} class to all tracks.', log=log)
        final_assignments = {}
        track_class_dict = {}

    ## Step 4: Save the results in in class_dict
    class_to_idx = {cls_name: idx+1 for idx, cls_name in enumerate(classes)}
    track_ids = []
    for dets_per_frame in class_dict['detections']:
        for det in dets_per_frame:
            track_id = det['track_id']
            assigned_class, assigned_score = final_assignments.get(track_id,(noid_str, 0.0))
            track_ids.append(track_id)
            # Save initial detection 
            det['det_id'] = det.get('category_id', 1)
            det['det_score'] = det.get('score')
            # Reassign the class and score based on the classification results
            det['category_id'] = class_to_idx[assigned_class]
            det['score'] = assigned_score
            # Save the extra keys for visua and analysis
            if track_id in track_class_dict:
                # Save the extra bbox coordinates if available
                if 'extra_bboxes' in track_class_dict[track_id]:
                    extra_bbox = track_class_dict[track_id]['extra_bboxes'].get(det['image_id']-1)
                    if extra_bbox is not None:
                        # Normalize the extra bbox coordinates to be in the range [0, 1] relative to the image size
                        det['extra_bbox'] = [
                            float(extra_bbox[0] / image_size[0]),
                            float(extra_bbox[1] / image_size[1]),
                            float(extra_bbox[2] / image_size[0]),
                            float(extra_bbox[3] / image_size[1])
                        ]
                # Save the path to the reference image if available
                if 'paths_ref' in track_class_dict[track_id] and assigned_class in track_class_dict[track_id]['paths_ref']:
                    det['path_ref'] = track_class_dict[track_id]['paths_ref'][assigned_class]
                # Save the path to the feature file if available
                if 'paths_crop' in track_class_dict[track_id] and assigned_class in track_class_dict[track_id]['paths_crop']:
                    det['path_crop'] = track_class_dict[track_id]['paths_crop'][assigned_class]
                # Save other scores and paths for other classes if available
                if 'scores' in track_class_dict[track_id]:
                    det['class_scores'] = {
                        cls_name: (
                            float(score),
                            track_class_dict[track_id]['paths_crop'][cls_name],
                            track_class_dict[track_id]['paths_ref'][cls_name]
                        )
                        for cls_name, score in track_class_dict[track_id]['scores'].items() if assigned_class != cls_name}
            
    track_ids = list(set(track_ids))
    n_tracks = len(track_ids)
    # Saving
    # class_dict['detection_classes'] = [f"Track {i}" for i in range(1, n_tracks + 1)]
    class_dict['classification_classes'] = classes
    class_dict['track_ids'] = track_ids
    save_json_file(class_dict, output_file)
    save_json_file(class_dict, output_file.replace('.json', '.pretty.json'), pretty=True)
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
    # Chrono
    start_time = time.time()

    # Video object initialization
    ## Check if input_video is already a VideoFrameIterator object
    if isinstance(args.input_video, VideoFrameIterator):
        my_video = args.input_video
    else:
        if not os.path.exists(args.input_video):
            print_and_log('Error: input %s must be a file or a folder' % (args.input_video), log=log, to_print=False)
            raise ValueError('No input provided.')
        my_video = VideoFrameIterator(args.input_video, log=log)
    if len(my_video) == 0:
        print_and_log('Input %s is empty. Skipping.' % (args.input_video), log=log, to_print=False)
        return 0
    image_size = my_video.get_image_size()
    print_and_log('Video %s opened with resolution %s and %d frames.' % (my_video.path, str(image_size), len(my_video)), log=log)

    # Load ground truth if evaluation is enabled
    if (args.eval_detection or args.eval_tracking or args.eval_classification):
        # If Gt file not provided, try to find it using video name + .zip or + _MOT.zip
        if gt_file_class_mot is None:
            gt_file_class_mot = find_file_with_ending(my_video.path[:-4], ['.zip'])
        if gt_file_class_mot is None:
            print_and_log('No ground truth file found for evaluation. Skipping evaluation.', log=log)
        else:
            # With class ID being the track id but also the det ID
            # boundaries = extract_boundaries(gt_file_class_mot, log=log)
            boundaries = None
            gt_dict_mot_cat, gt_labels = load_mot_format(gt_file_class_mot, boundaries=boundaries)

    # Detection
    os.makedirs(args.output, exist_ok=True)
    noid_str = 'NoID'
    if check_stop(log=log): return 0
    det_name = '%s%s' % (os.path.basename(args.det_model).split('.')[0], ("_" + args.text_prompt.replace(" ", "_")) if 'sam3' in args.det_model else "")
    detection_dict = detect(
        my_video,
        os.path.join(args.output, 'det_dicts', '%s.json' % det_name),
        device=args.device,
        score=args.det_score_th,
        tracking_size=args.tracking_size,
        det_model=args.det_model,
        text_prompt=args.text_prompt,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
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
        save_json_file(gt_dict_coco_det, gt_det_coco_file.replace('.json', '.pretty.json'), pretty=True)
        det_coco_file = save_coco_format(
            detection_dict['detections'],
            os.path.join(args.output, 'det_coco_format', det_name),
            image_size=image_size,
            labels=labels_detection,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(
            gt_det_coco_file,
            det_coco_file,
            name='%s%s' % (det_name, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            save_path=os.path.join(args.output, 'det_eval.csv'),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
            log=log
        )
        print_and_log('Detection evaluation results: %s' % (str(eval_results)), log=log)

    # Tracking
    if check_stop(log=log): return 0
    track_name = '%s_%s' % (det_name, args.tracker_type)
    tracking_dict = track(
        my_video,
        detection_dict,
        os.path.join(args.output, 'track_dicts', '%s.json' % track_name),
        device=args.device,
        tracking_size=args.tracking_size,
        score_th=args.det_score_th,
        tracker_type=None if args.tracker_type == 'IoU' and 'sam3' not in args.det_model else args.tracker_type,
        image_size=image_size,
        default_cat_id=1,
        log=log
    )

    if args.eval_tracking and gt_file_class_mot:
        # Load tracking results if not already loaded
        tracking_dict = load_as_detection_dict(tracking_dict, image_size=image_size, log=log)
        # Save tracking results in MOT format for evaluation
        track_mot_file = save_mot_format(
            tracking_dict['detections'],
            os.path.join(args.output, 'track_mot_format', track_name),
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
            name='%s%s' % (track_name, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            save_path=os.path.join(args.output, 'track_eval.csv'),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
        )
        print_and_log('Tracking evaluation results:\n%s' % (str(eval_results)), log=log)
    
    # Classification
    if check_stop(log=log): return 0
    classi_name = track_name
    classi_name += '_roi-%g' % (args.roi_factor)
    source_roi = classi_name
    classi_name += '_%s-thr-%g-nms-%g-roi-%g' % (args.class_det, args.class_det_thr, args.class_nms_thr, args.roi_det) if args.class_det else ''
    classi_name += '_featavg' if args.feat_avg else ''
    classi_name += '_nca_%d-%g' % (args.epochs, args.lr) if args.nca else ''
    classi_name += '_joint-%g' % (args.joint_factor) if args.joint_factor else ''
    classi_name += '_avg_score' if args.avg_score else ''
    classi_name += '_simth-%g' % (args.sim_th)
    class_dict = classify(
        tracking_dict,
        my_video,
        os.path.join(args.output, 'class_dicts', '%s.json' % (classi_name)),
        class_database=os.path.normpath(args.class_database),
        image_size=image_size,
        device=args.device,
        class_det=args.class_det,
        class_det_thr=args.class_det_thr,
        class_nms_thr=args.class_nms_thr,
        feat_avg=args.feat_avg,
        nca=args.nca,
        epochs=args.epochs,
        lr=args.lr,
        roi_factor=args.roi_factor,
        roi_det=args.roi_det,
        avg_score=args.avg_score,
        sim_th=args.sim_th,
        noid_str=noid_str,
        source_roi=source_roi,
        joint_factor=args.joint_factor,
        log=log
    )
    if check_stop(log=log): return 0
    classes = class_dict['classification_classes']
    
    if args.eval_classification and gt_file_class_mot:
        # Evaluate classification results using COCO metrics
        gt_dict_coco_class = mot_gt_to_coco_gt(gt_dict_mot_cat, image_size=image_size, categories=gt_labels)
        gt_class_coco_file = os.path.join(args.output, 'gt_class_coco_format.json')
        save_json_file(gt_dict_coco_class, gt_class_coco_file)
        save_json_file(gt_dict_coco_class, gt_class_coco_file.replace('.json', '.pretty.json'), pretty=True)
        gt_classes = [cat['name'] for cat in gt_dict_coco_class['categories']]
        # gt_class_coco_file = save_coco_format(gt_dict_coco_class['annotations'], os.path.join(args.output, 'gt_class_coco_format'), labels=gt_labels)
        uniform_class_list = solve_id_conflicts(class_dict['detections'], classes, gt_labels, default_label=noid_str, log=log)
        class_coco_file = save_coco_format(
            uniform_class_list,
            os.path.join(args.output, 'class_coco_format', classi_name),
            image_size=image_size,
            labels=classes,
            boundaries=boundaries
        )
        eval_results = evaluate_detection(
            gt_class_coco_file,
            class_coco_file,
            save_path=os.path.join(args.output, 'class_eval.csv'),
            name='%s%s' % (classi_name, ("_b" + "-".join(str(x) for pair in boundaries for x in pair)) if boundaries else ""),
            extra_info={
                "video_length": len(my_video),
                "resolution": "%dx%d" % (image_size[0], image_size[1])
            },
            ignore_classes=[gt['id'] for gt in gt_dict_coco_class['categories'] if gt['name'] == noid_str],
            log=log
        )
        print_and_log('Classification evaluation results: %s' % (str(eval_results)), log=log)

    # Save MOT format
    if args.save_mot:
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
            os.path.join(args.output, 'video_demos', '%s.mp4' % (classi_name)),
            max_res=args.max_res,
            display_fct=args.display_fct,
            detection_classes=class_dict.get('detection_classes', [args.text_prompt] if 'sam3' in args.det_model else None),
            classification_classes=class_dict.get('classification_classes'),
            track_ids=class_dict.get('track_ids', [0]),
            del_imgs=args.del_imgs,
            gt_annotations=coco_to_perso_format(gt_dict_coco_class['annotations'], image_size=image_size) if args.eval_classification and gt_file_class_mot else None,
            gt_classes=gt_classes if args.eval_classification and gt_file_class_mot else None,
            log=log
        )
        if check_stop(log=log): return 0

    print_and_log("Processing of %s finished in %ds." % (my_video.path, time.time()-start_time), log=log)
    return 1

def main_loop(args, log=None):
    '''
    Main loop to process the video. It allows to loop over the video and process it multiple times
    using different combination of parameters.

    Args:
        args: argparse.Namespace, the arguments
        log: logger, the logger to print the information
    '''
    testing = False
    if testing:
        det_models = ['sam3']
        prompts = ['a baboon']
        tracker_types = ['sam3']
        joint_factors = [0, 0.25, 0.5, 0.75]
        class_det_types = ['primateface', '']
        feat_avg = [False]
        nca = [True]
        epochs = [200]
        lr = [1e-4]
        roi_factors = [1.0]
        roi_dets = [2.5]
        avg_scores = [False, True]
        sim_ths = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        det_models = ['sam3', 'MDv5a', 'MDv5b', 'sam3_det']
        # prompts = ['a baboon', 'an animal', 'a monkey', 'a primate', 'an ape']
        prompts = ['a baboon', 'an animal']
        tracker_types = ['sam3', 'IoU', 'botsort', 'bytetrack', 'deepsort']
        joint_factors = [0, 0.25, 0.5, 0.75]
        class_det_types = ['primateface', '']
        # feat_avg = [True, False]
        feat_avg = [False]
        # nca = [True, False]
        nca = [True]
        epochs = [200]
        lr = [1e-4]
        roi_factors = [1.0, 1.25]
        roi_dets = [1, 1.8, 2.5]
        avg_scores = [False, True]
        sim_ths = [0, 0.5, 0.7]
    args.input_video = VideoFrameIterator(args.input_video, log=log)
    for det_model in det_models:
        args.det_model = det_model
        for tracker_type in tracker_types:
            if tracker_type == 'sam3' and det_model != 'sam3':
                continue
            args.tracker_type = tracker_type
            for prompt in prompts if 'sam3' in det_model else ['']:
                args.text_prompt = prompt
                for joint_factor in joint_factors:
                    args.joint_factor = joint_factor
                    for class_det in class_det_types if not joint_factor else ['']:
                        args.class_det = class_det
                        for feat in feat_avg if not joint_factor else [False]:
                            args.feat_avg = feat
                            for roi_factor in roi_factors:
                                args.roi_factor = roi_factor
                                for nca_val in nca if not joint_factor else [False]:
                                    args.nca = nca_val
                                    for epoch in epochs if nca_val else [0]:
                                        args.epochs = epoch
                                        for lr_val in lr if nca_val else [0]:
                                            args.lr = lr_val
                                            for roi_det in roi_dets if nca_val else [1.0]:
                                                args.roi_det = roi_det
                                                for avg_score in avg_scores:
                                                    args.avg_score = avg_score
                                                    for sim_th in sim_ths:
                                                        args.sim_th = sim_th
                                                        print_and_log('Running det %s%s and tracker %s%s' % (
                                                            det_model,
                                                            ' with prompt "%s"' % (prompt) if prompt else '',
                                                            tracker_type,
                                                            ' with classification%s%s%s%s' % (
                                                                ' with %s' % (class_det) if class_det else '',
                                                                ' with feat avg' if feat else '',
                                                                ' with NCA using epochs=%d, lr=%.0e, ROI det=%.2g' % (args.epochs, args.lr, args.roi_det) if nca_val else '',
                                                                ' with ROI factor %.2f' % (roi_factor) if roi_factor != 1.0 else ''
                                                            )
                                                        ), log=log)
                                                        main(args, log=log)
                                                        args.input_video.reset_video()

def final_evaluation(args, main_output, log=None):
    '''
    Perform a final evaluation on all the videos together if ground truth is available.

    Args:
        args: argparse.Namespace, the arguments
        main_output: str, the path to the main output folder
        log: logger, the logger to print the information
    '''
    eval_folder = os.path.join(main_output, 'final_evaluation')
    os.makedirs(eval_folder, exist_ok=True)
    video_outputs = sorted([os.path.join(main_output, f) for f in os.listdir(main_output) if os.path.isdir(os.path.join(main_output, f))])
    if args.eval_detection:
        # Evaluate detection results
        start_time = time.time()
        eval_file = os.path.join(eval_folder, 'det_eval.csv')
        print_and_log('Performing final detection evaluation on all videos together...', log=log)
        merge_eval_coco(
            video_outputs,
            eval_file,
            'gt_det_coco_format.json',
            'det_coco_format',
            os.path.join(eval_folder, 'det_eval'),
            log=log
        )
        print_and_log('Final detection evaluation results performed in %ds and saved in %s' % (time.time() - start_time, eval_file), log=log)

    if args.eval_tracking:
        # Evaluate tracking results
        start_time = time.time()
        print_and_log('Performing final tracking evaluation on all videos together...', log=log)
        eval_file = os.path.join(eval_folder, 'track_eval.csv')
        gt_file_per_video = {}
        pred_files_per_method_per_video = {}
        for video_output in video_outputs:
            gt_file = os.path.join(video_output, 'gt_track_mot_format')
            pred_root = os.path.join(video_output, 'track_mot_format')
            if not os.path.exists(gt_file) or 'mot.zip' not in os.listdir(gt_file):
                print_and_log('No ground truth file found for video %s. Skipping evaluation for this video.' % (video_output), log=log)
                continue
            gt_file_per_video[video_output] = gt_file
            for method_name in os.listdir(pred_root):
                if not os.path.isfile(os.path.join(pred_root, method_name, 'mot.zip')):
                    print_and_log('No prediction file found for method %s in video %s. Skipping evaluation for this method and video.' % (method_name, video_output), log=log)
                    continue
                if method_name not in pred_files_per_method_per_video:
                    pred_files_per_method_per_video[method_name] = {}
                pred_files_per_method_per_video[method_name][video_output] = os.path.join(pred_root, method_name)
        video_keys = list(gt_file_per_video.keys())
        gt_files = [gt_file_per_video[video_key] for video_key in video_keys]
        for method_name in pred_files_per_method_per_video:
            if len(pred_files_per_method_per_video[method_name]) != len(gt_file_per_video):
                print_and_log('Method %s does not have predictions for all videos. Skipping evaluation for this method.' % (method_name), log=log)
                continue
            pred_folders = [pred_files_per_method_per_video[method_name][video_key] for video_key in video_keys]
            merge_folder = os.path.join(eval_folder, 'track_mot_format', method_name)
            gt_merge, pred_merge = merge_mot_formats(gt_files, pred_folders, merge_folder)
            eval_results = evaluate_tracking(
                gt_merge,
                pred_merge,
                name=method_name,
                save_path=eval_file,
            )
            print_and_log('\tMethod %s: %s' % (method_name, str(eval_results)), log=log)

        print_and_log('Final tracking evaluation results performed in %ds and saved in %s' % (time.time() - start_time, eval_file), log=log)

    if args.eval_classification:
        # Evaluate classification results
        start_time = time.time()
        eval_file = os.path.join(eval_folder, 'class_eval.csv')
        merge_eval_coco(
            video_outputs,
            eval_file,
            'gt_class_coco_format.json',
            'class_coco_format',
            os.path.join(eval_folder, 'class_eval'),
            ignore_noid=True,
            cm=True,
            pr=True,
            visu=True,
            log=log
        )
        print_and_log('Final classification evaluation results performed in %ds and saved in %s' % (time.time() - start_time, eval_file), log=log)

def _process_video(args, input_path, main_output, main_funct, log_file=None):
    args = copy.deepcopy(args)
    args.input_video = input_path
    args.output = os.path.join(main_output, os.path.basename(input_path).split('.')[0])
    if log_file:
        log = setup_logger(log_file=log_file)
    else:
        log = None
    main_funct(args, log=log)
    close_log(log)

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
        log_file = os.path.join(args.output, 'logs', '%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
        log = setup_logger(log_file=log_file)
        print_and_log('Starting BaboonTrack without GUI with arguments: %s' % (str(vars(args))), log=log)
        # Check if input is video or has frames. If not, loop over the folder and process each video or set of frames separately.
        if os.path.isfile(args.input_video) or (os.path.isdir(args.input_video) and any(os.path.isfile(os.path.join(args.input_video, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(args.input_video))):
            main_funct(args, log=log)
        else:
            main_output = copy.deepcopy(args.output)
            input_list = sorted([os.path.join(args.input_video, f) for f in os.listdir(args.input_video)])
            if args.num_workers:
                args.parser = None
                import multiprocessing as mp
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
                    futures = [executor.submit(
                        _process_video,
                        args,
                        input_path,
                        main_output,
                        main_funct,
                        log_file.replace('.log', '_%d_%s.log' % (idx, os.path.basename(input_path)))) for idx, input_path in enumerate(input_list)]
                # propagate exceptions
                for f in futures:
                    f.result()
            else:
                for input_path in input_list:
                    _process_video(args, input_path ,main_output, main_funct, log=log)
            # In folder case, perform a final evaluation on all the videos together if ground truth is available
            final_evaluation(args, main_output, log=log)
        close_log(log)
    # Finish
    print('BaboonTrack finished.')
    return 1
