from argparse import ArgumentParser
import numpy as np
import os
import torch
import gc
import sys
import pdb
from pycocotools import mask as mask_utils
from json_utils import save_json_file
sys.path.append(os.path.dirname(__file__))  # Add sam3 directory to the system path to import sam3
from sam3.model_builder import build_sam3_video_predictor # type: ignore

'''
Miscellaneous functions
'''
def get_value_with_precision(value, precision=1000):
    '''
    Get the value with a certain precision.
    Useful for saving in json files (or other) to avoid float precision issues.
    
    Args:
        value: float, the value to process
        precision: int, the precision to use (default 1000)
    
    Returns:
        float: the value with the precision
    '''
    if type(value) is list:
        return np.trunc(precision*np.array(value))/precision
    elif value is None:
        return None
    else:
        return np.trunc(precision*value)/precision

def to_coco_format(sam_output, coco_list, frame_shift=0):
    frame_index = sam_output["frame_index"]+frame_shift
    track_ids = sam_output["outputs"]["out_obj_ids"]
    scores = sam_output["outputs"]["out_probs"]
    bboxes = sam_output["outputs"]["out_boxes_xywh"]  # Shape: [num_objects, 4] in percentage format (x, y, width, height)
    masks = sam_output["outputs"]["out_binary_masks"]  # Shape: [num_objects, height, width]
    for track_id, score, bbox, mask in zip(track_ids, scores, bboxes, masks):
        mask = mask.astype(np.uint8)  # Convert boolean mask to uint8 (0 and 1)
        rle = mask_utils.encode(np.asfortranarray(mask))  # Encode the binary mask using RLE
        rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string for JSON serialization
        coco_output = {
            "id": (len(coco_list) + 1),  # Unique ID for each detection
            "image_id": frame_index + 1,
            "category_id": 1,  # Assuming a single category for simplicity
            "track_id": track_id,
            "score": get_value_with_precision(score),
            "bbox": get_value_with_precision([bbox[0]*mask.shape[1], bbox[1]*mask.shape[0], bbox[2]*mask.shape[1], bbox[3]*mask.shape[0]], 10),  # Convert to absolute pixel values
            "segmentation": rle,   # Convert to list for JSON serialization
            "area": int(np.sum(mask)),  # Area of the mask (number of pixels)
            "iscrowd": 0  # Assuming all instances are not crowd
        }
        coco_list.append(coco_output)

def propagate_in_video(predictor, session_id, coco_list, frame_shift=0, cache_size=2, clear_freq=50):
    # we will just propagate from frame 0 to the end of the video
    for idx, response in enumerate(predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id
        )
    )):
        to_coco_format(response, coco_list, frame_shift=frame_shift)
        if idx >= cache_size:
            del predictor._all_inference_states[session_id]["state"]["cached_frame_outputs"][idx-cache_size]
        if idx % clear_freq == 0:
            gc.collect()
            torch.cuda.empty_cache()

def main(video_path, output_file, text_prompt, frame_shift=0):
    coco_list = []
    with torch.inference_mode():
        # Initialization
        video_predictor = build_sam3_video_predictor()
        # Start a session
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt,
            )
        )
        gpu_id = torch.cuda.current_device()
        gpu_tot = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print('Propagating in video. GPU memory (used/res/total): %.2f/%.2f/%.2f Gb' % (
            torch.cuda.memory_allocated(gpu_id) / 1024**3,
            torch.cuda.memory_reserved(gpu_id) / 1024**3,
            gpu_tot
        ))
        # Propagate the prompt in the video
        propagate_in_video(
            video_predictor,
            session_id,
            coco_list,
            frame_shift=frame_shift
        )
        # End the session
        video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    # Save the coco list to a json file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_json_file(coco_list, output_file)


def get_args():
    '''
    Get the arguments from the command line, process them and return them.

    Returns:
        args: argparse.Namespace, the arguments
    '''
    parser = ArgumentParser(description="Process thermal images to compute landmarks and signals.")
    parser.add_argument(
        '-i', '--video_path',
        type=str,
        help='Folder containing the images to process or a video file.'
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        help='Path to the output json file where the coco list will be saved.'
    )
    parser.add_argument(
        '-t', '--text_prompt',
        default='a Baboon',
        type=str,
        help='Text prompt to use for the SAM model to detect the objects of interest in the video.'
    )
    parser.add_argument(
        '-f', '--frame_shift',
        type=int,
        default=0,
        help='Shift for saving the image_idx.'
    )
    args = parser.parse_args()
    args.parser = parser
    return args

if __name__ == "__main__":
    args = get_args()
    main(args.video_path, args.output_file, args.text_prompt, frame_shift=args.frame_shift)