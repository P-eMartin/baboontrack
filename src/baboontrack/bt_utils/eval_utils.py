from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
import numpy as np
import copy
import json
import os
from .io_utils import print_and_log, save_mot_format, save_json_file, save_coco_format, load_mot_format, mot_to_coco_format
from .bot_sort.tracking_utils.evaluation import Evaluator

## Perform evaluation of the detection and tracking results

def evaluate_detection(gt_file, detection_file, log=None):
    '''
    Evaluate the detection performance using COCO metrics.

    Args:
        gt_file: str, path to the ground truth JSON file or zip file in COCO format
        detection_file: str, path to the detection JSON file or zip file in COCO format
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the evaluation results
    '''
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(detection_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_results = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
    }
    return eval_results

def evaluate_tracking(gt_file, tracking_file, log=None):
    '''
    Evaluate the tracking performance using MOT metrics.

    Args:
        gt_file: str, path to the ground txt file or zip file in MOT format
        tracking_file: str, path to the tracking results file in MOT format
        log: logger, the logger to print the information (default None)

    Returns:
        dict, the evaluation results
    '''
    # os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
    #  __init__(self, data_root, seq_name, data_type)
    data_root = os.path.dirname(gt_file)
    seq_name = os.path.basename(gt_file).split('.')[0]
    evaluator = Evaluator(data_root, seq_name, data_type='mot')
    eval_results = evaluator.eval_file(tracking_file)
    return eval_results

