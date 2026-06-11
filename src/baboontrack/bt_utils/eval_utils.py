import pdb

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pandas as pd
# To avoid the error "AttributeError: module 'numpy' has no attribute 'asfarray'" when using motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, **kwargs: np.asarray(a, dtype=float)
import csv
import os
from .io_utils import print_and_log, save_json_file, get_value_with_precision, save_dict_as_csv, zip_folder
from collections import defaultdict
import zipfile
from .bot_sort.tracking_utils.evaluation import Evaluator

'''
Saving formats
'''
def coco_to_perso_format(coco_list, image_size=None, frame_id_offset=0):
    '''
    Convert a COCO-format list of annotations to a custom format.

    Args:
        coco_list: list of dict, the COCO-format list of annotations
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        frame_id_offset: int, the offset to add to the frame IDs (default 0, to keep the same frame IDs)

    Returns:
        list of dict, the custom-format list of annotations
    '''
    perso_list = [[] for i in range(frame_id_offset)] # Initialize the perso_list with empty lists for the frame ID offset
    det_list = []
    prev_img_id = 1
    for det in coco_list:
        img_id = det['image_id']
        if prev_img_id != img_id:
            perso_list.append(det_list)
            det_list = []
            # Add empty lists for the frames between the previous image ID and the current image ID
            for _ in range(img_id - prev_img_id - 1):
                perso_list.append([])
            prev_img_id = img_id
        det_list.append({
            'image_id': det['image_id'] + frame_id_offset,
            'category_id': det['category_id'],
            'bbox': det['bbox'] if image_size is None else [
                det['bbox'][0] / image_size[0],
                det['bbox'][1] / image_size[1],
                det['bbox'][2] / image_size[0],
                det['bbox'][3] / image_size[1]
            ],
            'score': det.get('score', det.get('det_score', 1)),
            'track_id': det.get('track_id', det.get('attributes', {}).get('track_id')),
            # Add segmentation if available
            'segmentation': det.get('segmentation'),
        })
    perso_list.append(det_list)
    return perso_list

def perso_format_to_trackid_format(perso_list):
    '''
    Convert a custom-format list of annotations to a track ID format.

    Args:
        perso_list: list of list of dict, the custom-format list of annotations

    Returns:
        dict, key: track ID, value: list of annotations
    '''
    trackid_dict = defaultdict(list)
    for frame_dets in perso_list:
        for det in frame_dets:
            trackid_dict[det['track_id']].append(det)
    return dict(trackid_dict)

def save_coco_format(detection_dict, output_path, image_size=None, labels=None, boundaries=None, frame_id_offset=0):
    '''
    Save the detection and tracking results in COCO format.

    Args:
        detection_dict: dict, the detection and tracking results
        output_path: str, the path to save the output file
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        labels: list of str, the list of labels to save in a separate file (default None)
        boundaries: list of int, the boundaries of the frames to save (default None)
        frame_id_offset: int, the offset to add to the frame IDs (default 0, no offset)

    Returns:
        int, 1 if the file was properly saved
    '''
    # Create the COCO format dictionary
    coco_list = []
    # Loop over the detection_dict and fill the COCO format dictionary
    for idx, dets_or_key in enumerate(detection_dict):
        if isinstance(dets_or_key, list): # Case when dets_or_key is a list of detections
            if boundaries and not (boundaries[0] <= idx <= boundaries[1]):
                continue
            for det in dets_or_key:
                coco_list.append({
                    'image_id': idx + 1 + frame_id_offset,
                    'category_id': int(det['id'] + 1) if 'id' in det else 1,
                    'bbox': [
                        get_value_with_precision(det['bbox'][0] * image_size[0] if image_size is not None else det['bbox'][0], 10),
                        get_value_with_precision(det['bbox'][1] * image_size[1] if image_size is not None else det['bbox'][1], 10),
                        get_value_with_precision(det['bbox'][2] * image_size[0] if image_size is not None else det['bbox'][2], 10),
                        get_value_with_precision(det['bbox'][3] * image_size[1] if image_size is not None else det['bbox'][3], 10)
                    ],
                    'score': get_value_with_precision(det.get('det_score', det.get('score', 1))),
                    'attributes': {
                        'name': 'NoID',
                        'track_id': int(det['track_id'] + 1),
                        'visibility': det.get('visibility', 1),
                    }
                })
        else: # Case when detection_dict is a list of dictionaries (already in COCO format)
            det = dets_or_key
            if boundaries and not (boundaries[0] <= det['image_id']-1 <= boundaries[1]):
                continue
            coco_list.append({
                'image_id': det['image_id'] + frame_id_offset,
                'category_id': det['id'] if 'id' in det else 1,
                'bbox': [
                    get_value_with_precision(det['bbox'][0] * image_size[0], 10) if image_size is not None else det['bbox'][0],
                    get_value_with_precision(det['bbox'][1] * image_size[1], 10) if image_size is not None else det['bbox'][1],
                    get_value_with_precision(det['bbox'][2] * image_size[0], 10) if image_size is not None else det['bbox'][2],
                    get_value_with_precision(det['bbox'][3] * image_size[1], 10) if image_size is not None else det['bbox'][3]
                ],
                'score': get_value_with_precision(det['score'] if 'score' in det else 1),
                'area': det['area'] if 'area' in det else get_value_with_precision(det['bbox'][2] * det['bbox'][3] if image_size is None else det['bbox'][2] * image_size[0] * det['bbox'][3] * image_size[1], 10),
                'iscrowd': det['iscrowd'] if 'iscrowd' in det else 0,
                'attributes': {
                    'name': 'NoID',
                    'track_id': det['track_id'],
                    'visibility': det['visibility'] if 'visibility' in det else 1,
                }
            })
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'detections.json')
    save_json_file(coco_list, output_file)
    # Save labels if provided
    if labels is not None:
        save_json_file(labels, os.path.join(output_path, 'labels.json'))
    return output_file

def save_mot_format(detection_dict, output_path, image_size=None, labels=None, boundaries=None, cat_id_override=None, frame_id_offset=1):
    '''
    Save the detection and tracking results in MOT format.

    Args:
        detection_dict: dict, the detection and tracking results
        output_path: str, the path to save the output files
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        labels: list of str, the list of labels to save in a separate file (default None)
        boundaries: list of int, the boundaries of the frames to save (default None)
        cat_id_override: int, the category ID to use as override for all annotations (default None, if None, the provided category ID is used)
        frame_id_offset: int, the offset to add to the frame IDs (default 1, to start from 1 instead of 0)

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
    for idx, dets_or_key in enumerate(detection_dict):
        if isinstance(dets_or_key, list): # Case when dets_or_key is a list of detections
            if boundaries and not (boundaries[0] <= idx <= boundaries[1]):
                continue
            for det in dets_or_key:
                mot_dict['frame_id'].append(idx+frame_id_offset)
                mot_dict['track_id'].append(det['track_id']+1)
                mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
                mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
                mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
                mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
                mot_dict['not ignored'].append(1)
                mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else int(det.get('id', 0)+1))
                mot_dict['visibility'].append(det.get('visibility', 1))
                mot_dict['skipped'].append(0)
        elif isinstance(detection_dict, dict): # Case when detection_dict is a dictionary with dets_or_key being the key.
            if boundaries and not (boundaries[0] <= int(dets_or_key)-1 <= boundaries[1]):
                continue
            dets = detection_dict[dets_or_key]
            for det in dets:
                mot_dict['frame_id'].append(dets_or_key + frame_id_offset -1)
                mot_dict['track_id'].append(det['track_id'])
                mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
                mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
                mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
                mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
                mot_dict['not ignored'].append(det.get('not_ignored', 1))
                mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else int(det.get('id', 0)+1))
                mot_dict['visibility'].append(det.get('visibility', 1))
                mot_dict['skipped'].append(det.get('skipped', 0))
        else: # Case when detection_dict is a list of dictionaries (in COCO format)
            det = dets_or_key
            if boundaries and not (boundaries[0] <= det['image_id']-1 <= boundaries[1]):
                continue
            mot_dict['frame_id'].append(det['image_id'] + frame_id_offset -1)
            mot_dict['track_id'].append(det['track_id'])
            mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
            mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
            mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
            mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
            mot_dict['not ignored'].append(det.get('not_ignored', 1))
            mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else det.get('category_id', 1))
            mot_dict['visibility'].append(det.get('visibility', 1))
            mot_dict['skipped'].append(det.get('skipped', 0))
    output_file = os.path.join(folder_to_zip, 'gt.txt')
    save_dict_as_csv(mot_dict, output_file, without_headers=True)
    # Save labels if provided
    if labels is not None:
        with open(os.path.join(folder_to_zip, 'labels.txt'), 'w') as f:
            for label in labels:
                f.write('%s\n' % (label))
    zip_folder(folder_to_zip, os.path.join(output_path, 'mot.zip'))
    return output_file

def load_mot_format(input_file, boundaries=None, log=None):
    '''
    Load detection/tracking results from a MOT-format file.

    Args:
        input_file: str, the path to the input file (can be a zip file, a folder containing gt/gt.txt, or a gt.txt file)
        boundaries: tuple of int, the start and end frame IDs to load (default None, if None, all frames are loaded
        log: logging.Logger, the logger to log the information (default None)

    Returns:
        dict: a dictionary containing the loaded detection/tracking results, with frame IDs as keys and
            lists of detections as values. Each detection is a dictionary with keys 'track_id', 'bbox', 'id', 'visibility', 'not_ignored', and 'skipped'.
    '''

    def parse_rows(reader):
        detection_dict = defaultdict(list)

        for row in reader:
            frame_id = int(row[0])

            detection_dict[frame_id].append({
                "track_id": int(row[1]) - 1,
                "bbox": [
                    max(int(float(row[2])), 0),
                    max(int(float(row[3])), 0),
                    max(int(float(row[4])), 0),
                    max(int(float(row[5])), 0)
                ],
                "id": int(float(row[7])) - 1,
                "visibility": float(row[8]),
                "not_ignored": int(row[6]),
                "skipped": int(row[9]) if len(row) > 9 else 0
            })

        return dict(detection_dict)

    if zipfile.is_zipfile(input_file):
        with zipfile.ZipFile(input_file) as zf:
            with zf.open("gt/gt.txt") as f:
                reader = csv.reader(line.decode() for line in f)
                detection_dict = parse_rows(reader)

    elif os.path.isdir(input_file):
        with open(os.path.join(input_file, "gt", "gt.txt"), newline="") as f:
            detection_dict = parse_rows(csv.reader(f))

    elif os.path.isfile(input_file):
        with open(input_file, newline="") as f:
            detection_dict = parse_rows(csv.reader(f))

    else:
        raise ValueError(
            f"Input '{input_file}' is not a valid zip file, folder, or file."
        )

    if boundaries is not None:
        start, end = boundaries
        detection_dict = {
            frame_id: dets
            for frame_id, dets in detection_dict.items()
            if start <= frame_id-1 <= end
        }

    return detection_dict

def save_mot_to_mot(input_file, output_folder, boundaries=None, log=None):
    '''
    An utility based on the load_mot_format function to save MOT format files with the same or different boundaries.
    
    Args:
        input_file: str, the path to the input MOT format file
        output_folder: str, the path to the output folder which will contain the gt/gt.txt file
        boundaries: tuple of int, the start and end frame IDs to save (default None, if None, all frames are saved)
        log: logging.Logger, the logger to log the information (default None)

    Returns:
        int, 1 if the file was properly saved
    '''
    detection_dict = load_mot_format(input_file, boundaries=boundaries, log=log)

def mot_to_coco_format(mot_dict, image_size=None, cat_id_override=None):
    '''
    Convert a MOT-format dictionary to COCO format.

    Args:
        mot_dict: dict, the MOT-format dictionary to convert
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        cat_id_override: int, the category ID to use as override for all annotations (default None, if None, the provided category ID is used)

    Returns:
        list of dict, the converted COCO-format list of annotations
    '''
    coco_list = []
    idx = -1
    for frame_id, dets in mot_dict.items():
        for det in dets:
            idx += 1
            coco_list.append({
                'id': idx,
                'image_id': frame_id,
                'category_id': cat_id_override if cat_id_override is not None else int(det['id'] + 1) if 'id' in det else 1,
                'bbox': [
                    get_value_with_precision(det['bbox'][0] * image_size[0] if image_size is not None else det['bbox'][0], 10),
                    get_value_with_precision(det['bbox'][1] * image_size[1] if image_size is not None else det['bbox'][1], 10),
                    get_value_with_precision(det['bbox'][2] * image_size[0] if image_size is not None else det['bbox'][2], 10),
                    get_value_with_precision(det['bbox'][3] * image_size[1] if image_size is not None else det['bbox'][3], 10)
                ],
                'score': get_value_with_precision(det.get('score', det.get('det_score', 1))),
                'iscrowd': 0,
                'area': get_value_with_precision(det['bbox'][2] * det['bbox'][3] if image_size is None else det['bbox'][2] * image_size[0] * det['bbox'][3] * image_size[1], 10),
                'attributes': {
                    'name': 'NoID',
                    'track_id': int(det['track_id'] + 1),
                    'visibility': det['visibility'] if 'visibility' in det else 1,
                }
            })
    return coco_list

def mot_gt_to_coco_gt(mot_gt_file, image_size=None, cat_id_override=None, categories=None):
    '''
    Convert a MOT-format ground truth file to COCO format.

    Args:
        mot_gt_file: str or dict, the path to the MOT-format ground truth file (can be a zip file, a folder containing gt/gt.txt, or a gt.txt file)
        image_size: list of int, the size of the image in the format [width, height] (default None, if None, the bbox coordinates are not normalized)
        cat_id_override: int, the category ID to use as override for all annotations (default None, if None, the provided category ID is used)
        categories: list, the list of category names to include in the COCO output (default None)

    Returns:
        dict, the converted COCO-format ground truth dictionary with 'annotations' and 'categories'
    '''
    if isinstance(mot_gt_file, str):
        mot_dict = load_mot_format(mot_gt_file)
    else:
        mot_dict = mot_gt_file
    coco_gt = {
        'images': [{'id': frame_id, 'width': image_size[0] if image_size is not None else 1920, 'height': image_size[1] if image_size is not None else 1080} for frame_id in mot_dict.keys()],
        'annotations': mot_to_coco_format(mot_dict, cat_id_override=cat_id_override),
        'categories': [{'id': idx+1, 'name': cat} for idx, cat in enumerate(categories)] if categories is not None else [{'id': 1, 'name': 'NoID'}]
    }
    return coco_gt

def update_eval_csv(
    csv_path,
    segment_name,
    metrics,
    extra_info=None,
):
    """
    Update or append evaluation results in a CSV file.

    Args:
        csv_path: str
            Path to CSV file storing all evaluations.

        segment_name: str
            Unique identifier for the evaluated video/segment.

        metrics: dict
            COCO evaluation metrics.

        extra_info: dict (optional)
            Additional metadata (e.g., video length, FPS, model name).
    """

    # ------------------------------------------------------------
    # 1. Build row dictionary
    # ------------------------------------------------------------
    row = {"segment_name": segment_name}

    # add metrics
    row.update(metrics)

    # add optional metadata
    if extra_info is not None:
        row.update(extra_info)

    # ------------------------------------------------------------
    # 2. Load existing CSV if it exists
    # ------------------------------------------------------------
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # ------------------------------------------------------------
    # 3. Update or append row
    # ------------------------------------------------------------
    if "segment_name" in df.columns and segment_name in df["segment_name"].values:

        # overwrite existing row
        df.loc[df["segment_name"] == segment_name, row.keys()] = row.values()

    else:
        # append new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # ------------------------------------------------------------
    # 4. Save back
    # ------------------------------------------------------------
    df.to_csv(csv_path, index=False)

#####
## Evaluation of the detection and tracking results
#####
def coco_eval(gt_file, detection_file):
    """
    Run COCO evaluation and return metrics as a flat dictionary.
    """

    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(detection_file)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR": coco_eval.stats[6],
        "AR50": coco_eval.stats[7],
        "AR75": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

def evaluate_detection(gt_file, detection_file, name=None, save_path=None, extra_info=None):
    '''
    Evaluate the detection performance using COCO metrics.

    Args:
        gt_file: str, path to the ground truth JSON file or zip file in COCO format
        detection_file: str, path to the detection JSON file or zip file in COCO format
        save_path: str, path to save the evaluation results (default None)
        extra_info: dict (optional), additional metadata to include in the evaluation results

    Returns:
        dict, the evaluation results
    '''
    metrics = coco_eval(gt_file, detection_file)
    if save_path is not None:
        update_eval_csv(
            csv_path=save_path,
            segment_name=name if name is not None else os.path.basename(detection_file).split('.')[0],
            metrics=metrics,
            extra_info=extra_info
        )
    return metrics

def mot_eval(gt_file, tracking_file):
    """
    Evaluate MOT tracking performance and return a flat dictionary.
    """

    data_root = os.path.dirname(gt_file)
    seq_name = os.path.basename(gt_file).split('.')[0]

    evaluator = Evaluator(data_root, seq_name, data_type="mot")

    eval_results = evaluator.eval_file(tracking_file)
    summary = evaluator.get_summary([eval_results], [seq_name])

    # ------------------------------------------------------------
    # Extract ONLY the OVERALL row (remove duplication noise)
    # ------------------------------------------------------------
    if "OVERALL" in summary.index:
        row = summary.loc["OVERALL"]
    else:
        row = summary.iloc[0]

    # convert to clean dict
    metrics = row.to_dict()

    # add identifier
    metrics["sequence"] = seq_name

    return metrics

def evaluate_tracking(gt_file, tracking_file, save_path=None, name=None, extra_info=None):
    '''
    Evaluate the tracking performance using MOT metrics.

    Args:
        gt_file: str, path to the ground txt file or zip file in MOT format
        tracking_file: str, path to the tracking results file in MOT format
        save_path: str, path to save the evaluation results (default None)
        name: str, name of the sequence (default None)
        extra_info: dict (optional), additional metadata to include in the evaluation results

    Returns:
        dict, the evaluation results
    '''
    metrics = mot_eval(gt_file, tracking_file)
    if save_path is not None:
        update_eval_csv(
            csv_path=save_path,
            segment_name=name if name is not None else os.path.basename(tracking_file).split('.')[0],
            metrics=metrics,
            extra_info=extra_info
        )
    return metrics

