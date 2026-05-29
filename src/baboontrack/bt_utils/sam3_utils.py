import torch
import pdb
import os
import numpy as np
import shutil
import psutil
import time
from .img_utils import VideoFrameIterator
from .io_utils import print_and_log, get_all_files_in_folder
from .json_utils import save_json_file, load_json_file
from .ffmpeg_utils import run_command
from pycocotools import mask as mask_utils
import networkx as nx
import copy
from collections import defaultdict

# def merge_detections(coco_files, iou_threshold=0.9):
#     '''
#     Merge the overlapped detections in coco_list that have a high IoU
#     and propagate the track_id of the merged detections.

#     Args:
#         coco_files: list of paths to COCO format files
#         iou_threshold: float, the IoU threshold to use for merging
#     '''
#     # Per frames
#     dets_per_frame = defaultdict(list)
#     for chunk_idx, coco_file in enumerate(coco_files):
#         for det in load_json_file(coco_file):
#             dets_per_frame[det['image_id']].append({"chunk": chunk_idx, "det": det})
#     # Find frames with overlaps
#     overlap_frames = []
#     for image_id, dets in dets_per_frame.items():
#         chunks_present = {d["chunk"] for d in dets}
#         if len(chunks_present) > 1:
#             overlap_frames.append(image_id)
#     # Tracks
#     tracks = defaultdict(list)
#     for chunk_idx, coco_file in enumerate(coco_files):
#         for det in load_json_file(coco_file):
#             key = (chunk_idx, det["track_id"])
#             tracks[key].append(det)


#     overlap_frame = []
#     overlaps_bol = [len(dets)>1 for dets in dets_per_frame.values()]
#     # Starts represent the starting of overlaps (consecutive frames with more than 1 detection), ends represent the end of overlaps
#     starts = [0] + [i+1 for i in range(len(overlaps_bol)-1) if overlaps_bol[i] and not overlaps_bol[i+1]]
#     ends = [i+1 for i in range(len(overlaps_bol)-1) if not overlaps_bol[i] and overlaps_bol[i+1]] + [len(overlaps_bol)]
    
#     # end = 0
#     # while end < len(coco_list):
#     #     end = min()
#     # for frame_detections in coco_list:
#     #     if len(frame_detections) <= 1:
#     #         continue
#     #     merged_detections = []
#     #     for det in frame_detections:

def compute_mask_iou(det1, det2):
    '''
    Compute IoU between two COCO detections using their segmentation masks.

    Args:
        det1: dict, first detection
        det2: dict, second detection

    Returns:
        iou: float, IoU between the two detections from mask
    '''
    rle1 = det1["segmentation"]
    rle2 = det2["segmentation"]
    return float(mask_utils.iou([rle1], [rle2], [0])[0][0])


def merge_detections(coco_files,iou_threshold=0.8):
    '''
    Merge detections coming from overlapping chunks.
        1) Load all detections from every chunk.
        2) Build track objects indexed by (chunk_idx, local_track_id).
        3) Find overlap frames shared by consecutive chunks.
        4) Compute track correspondences using average IoU on overlap frames.
        5) Build a graph of matching tracks.
        6) Connected components define global track IDs.
        7) Rewrite track IDs.
        8) Remove duplicated detections inside overlap frames.

    Args:
        coco_files: list, paths to COCO format files
        iou_threshold: float, the IoU threshold to use for merging tracks

    Returns:
        merged_detections: list of dicts, the merged detections with global track IDs
    '''
    # STEP 1 - LOAD DETECTIONS
    all_detections = []
    for chunk_idx, coco_file in enumerate(coco_files):
        detections = load_json_file(coco_file)
        for det in detections:
            det = copy.deepcopy(det)
            det["_chunk_idx"] = chunk_idx
            all_detections.append(det)

    # STEP 2 - BUILD TRACK STRUCTURE
    tracks = defaultdict(list)
    for det in all_detections:
        key = (det["_chunk_idx"], det["track_id"])
        tracks[key].append(det)

    # STEP 3 - INDEX TRACKS BY FRAME
    track_frames = {}
    for track_key, dets in tracks.items():
        frame_dict = {}
        for det in dets:
            frame_dict[det["image_id"]] = det
        track_frames[track_key] = frame_dict

    # STEP 4 - BUILD MATCHING GRAPH
    graph = nx.Graph()
    for track_key in tracks:
        graph.add_node(track_key)
    n_chunks = len(coco_files)

    ## Compare only consecutive chunks
    for chunk_idx in range(n_chunks - 1):
        current_tracks = [k for k in tracks.keys() if k[0] == chunk_idx]
        next_tracks = [k for k in tracks.keys() if k[0] == chunk_idx + 1]

        # Compute pairwise track similarity
        for track_a in current_tracks:
            frames_a = track_frames[track_a]
            for track_b in next_tracks:
                frames_b = track_frames[track_b]
                common_frames = (set(frames_a.keys()) & set(frames_b.keys()))

                if not common_frames:
                    continue

                ious = []
                for frame_id in common_frames:
                    iou = compute_mask_iou(frames_a[frame_id], frames_b[frame_id])
                    ious.append(iou)

                mean_iou = np.mean(ious)
                if mean_iou >= iou_threshold:
                    graph.add_edge(track_a, track_b, weight=float(mean_iou))

    # STEP 5 - ASSIGN GLOBAL TRACK IDS
    track_mapping = {}
    global_track_id = 0
    for component in nx.connected_components(graph):
        for local_track in component:
            track_mapping[local_track] = global_track_id
        global_track_id += 1

    # STEP 6 - REWRITE TRACK IDS
    rewritten_detections = []
    for det in all_detections:
        local_track = (det["_chunk_idx"], det["track_id"])
        det["track_id"] = track_mapping[local_track]
        rewritten_detections.append(det)

    # STEP 7 - REMOVE DUPLICATES
    detections_by_frame_track = defaultdict(list)
    for det in rewritten_detections:
        key = (det["image_id"], det["track_id"])
        detections_by_frame_track[key].append(det)

    merged_detections = []
    for key, dets in detections_by_frame_track.items():
        if len(dets) == 1:
            merged_detections.append(dets[0])
            continue
        # We keep the detection with the largest area.
        best_det = max(dets, key=lambda d: float(mask_utils.area(d["segmentation"])))
        merged_detections.append(best_det)

    # STEP 8 - CLEAN INTERNAL FIELDS
    for det in merged_detections:
        det.pop("_chunk_idx", None)
    merged_detections.sort(key=lambda d: (d["image_id"], d["track_id"]))
    return merged_detections

def process_video_with_sam(my_video, output_file, text_prompt="a Baboon", chunk_size=400, overlap=5, tmp_dir=".tmp", clean_up=False, log=None):
    # Initialization
    start_time = time.time()
    ## Frame extraction
    tmp_vid = os.path.join(tmp_dir, os.path.basename(my_video.path).split(".")[0])
    if os.path.exists(tmp_vid):
        if len(my_video) == len(get_all_files_in_folder(tmp_vid, extensions=(".jpg", ".png")))-overlap*(len(os.listdir(tmp_vid))-1):
            print_and_log(f"Frames already extracted in '{tmp_vid}'. Skipping extraction.", log=log)
        else:
            # Remove the existing folder and extract frames again
            print_and_log(f"Existing folder '{tmp_vid}' has an unexpected number of frames. Removing it and extracting frames again.", log=log)
            shutil.rmtree(tmp_vid)
            my_video.extract_all_frames_in_chunks(tmp_vid, chunk_size, overlap)
    else:
        my_video.extract_all_frames_in_chunks(tmp_vid, chunk_size, overlap)
    chunk_dirs = sorted([os.path.join(tmp_vid, f) for f in os.listdir(tmp_vid) if os.path.isdir(os.path.join(tmp_vid, f))])
    coco_files = []
    frame_shift = 0
    ram_tot = psutil.virtual_memory().total / 1024**3
    print_and_log("Processing video in %d chunks of %d frames with an overlap of %d frames." % (len(chunk_dirs), chunk_size, overlap), log=log)

    for chunk_dir in chunk_dirs:
        # Process each chunk in another script to avoid memory overload from the video predictor
        # conda run -n ENV_NAME python script.py
        coco_file = os.path.join(chunk_dir, 'coco_list.json')
        print_and_log(
            'Processing chunk %s with RAM %.2f/%.2f' % (
                chunk_dir,
                (ram_tot - psutil.virtual_memory().available/1024**3),
                ram_tot
            ),
            log=log
        )
        command = ['conda', 'run', '-n', 'sam3', 'python', os.path.join(os.path.dirname(__file__), 'sam3_run.py'),
                   '-i', chunk_dir, '-o', coco_file, '-t', text_prompt, '-f', str(frame_shift)]
        run_command(command, log=log)
        frame_shift += chunk_size - overlap
        if clean_up:
            shutil.rmtree(chunk_dir)
        coco_files.append(coco_file)

    # Merge coco files
    merged_detections = merge_detections(coco_files, iou_threshold=0.8)
    save_json_file(merged_detections, output_file)

    if clean_up:
        shutil.rmtree(tmp_vid)

    return merged_detections



