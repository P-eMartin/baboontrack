from argparse import ArgumentParser
import os
import datetime
import torch
from .help import *

def split_or_empty(string):
    '''
    Split string and return the last element if not empty, else return empty string.

    Args:
        string: string to split.

    Returns:
        string, last element of the split string if not empty, else empty string.
    '''
    return string if string == '' else os.path.splitext(os.path.basename(os.path.normpath(string)))[0]

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
        default='sam3',
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
        default='sam3',
        help=helptext_tracker_type
    )
    parser.add_argument(
        '-p', '--text_prompt',
        type=str,
        default="a baboon",
        help=helptext_text_prompt
    )
    parser.add_argument(
        '-k', '--chunk_size',
        type=int,
        default=200,
        help=helptext_chunk_size
    )
    parser.add_argument(
        '-O', '--overlap',
        type=int,
        default=5,
        help=helptext_overlap
    )
    parser.add_argument(
        '-P', '--class_database',
        default=os.path.join('/shared', 'group_dict_10-07'),
        type=str,
        help=helptext_class_database
    )
    parser.add_argument(
        '-Cdet', '--class_det',
        default=None,
        type=str,
        help=helptext_class_det
    )
    parser.add_argument(
        '-CdetThr', '--class_det_thr',
        default=0.5,
        type=float,
        help=helptext_class_det_thr
    )
    parser.add_argument(
        '-CnmsThr', '--class_nms_thr',
        default=0.4,
        type=float,
        help=helptext_class_nms_thr
    )
    parser.add_argument(
        '-f', '--feat_avg',
        action='store_true',
        help=helptext_feat_avg
    )
    parser.add_argument(
        '-n', '--nca',
        action='store_true',
        help=helptext_nca
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help=helptext_epochs
    )
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help=helptext_lr
    )
    parser.add_argument(
        '-F', '--roi_factor',
        default=1.0,
        type=float,
        help=helptext_roi_factor
    )
    parser.add_argument(
        '-R', '--roi_det',
        default=1.0,
        type=float,
        help=helptext_roi_det
    )
    parser.add_argument(
        '-a', '--avg_score',
        action='store_true',
        help=helptext_avg_score
    )
    parser.add_argument(
        '-S', '--sim_th',
        default=0.5,
        type=float,
        help=helptext_sim_th
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
    parser.add_argument(
        '-m', '--save_mot',
        action='store_true',
        help=helptext_save_mot
    )
    args = parser.parse_args()
    args.parser = parser
    infer_args_name(args)
    return args

def check_args(args=None, **kwargs):
    '''
    Check arguments and return them.

    Returns:
        args: argparse.Namespace, the arguments
    '''
    if args is None:
        args = get_args()
    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            args.parser.print_help()
            raise ValueError('Attribute %s not found in args.' % (key))
    args.display_fct = None
    return args

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