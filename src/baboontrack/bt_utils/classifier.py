from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import os
from .primateface import PrimateFace

class PrimateFaceDetector:
    def __init__(self, device='cuda', det_thr=0.5, nms_thr=0.4):
        pf = PrimateFace(
            device= device,
            pose_model = None,
            det_threshold = det_thr,
            nms_threshold = nms_thr
        )
        return pf
    
    def detect(self, img):
        bboxes, scores = self._processor.detect_primates(img)
        return bboxes, scores

def cosine_similarity(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()

def load_model_and_transform(model_path='', device='cpu'):
    # Load the model
    # model = torch.load(model_path, map_location=device)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    # Device
    model.to(device)
    model.eval()

    # Define the transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return model, transform


def extract_feature(model, transform, img):
    '''
    Extract the feature of an RGB image using the model and the transform.
    
    Args:
        model: torch.nn.Module, the model to use for feature extraction
        transform: torchvision.transforms, the transform to apply to the image
        img: str or numpy.ndarray, the path to the image or the image itself

    Returns:
        torch.Tensor, the extracted feature
    '''

    if isinstance(img, str):
        image = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        image = img.convert("RGB")
    else:
        raise ValueError("img should be a string (path to the image) or a numpy array (the image itself)")

    x = transform(image).unsqueeze(0)

    # Device
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        feat = model(x)

    feat = feat.squeeze()

    # normalize
    feat = feat / feat.norm()

    return feat

def build_image_paths_dict(class_dict_path):
    '''
    Build a dictionary of image paths for each class.
    
    Args:
        class_dict_path: str, the path to the classification dictionary, which should be a folder containing subfolders for each class, and each subfolder should contain the images corresponding to that class.

    Returns:
        image_paths: dict, a dictionary of image paths for each class.
    '''
    image_paths = {}
    for class_name in os.listdir(class_dict_path):
        class_path = os.path.join(class_dict_path, class_name)
        if os.path.isdir(class_path):
            image_paths[class_name] = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
    return image_paths

def build_feature_dict(model, transform, image_paths):
    '''
    Build a dictionary of features for a list of image paths.
    
    Args:
        model: torch.nn.Module, the model to use for feature extraction
        transform: torchvision.transforms, the transform to apply to the images
        image_paths: dict, a dictionary of list of image paths, with the keys being the IDs of the images (e.g. track IDs) and the values being the list of image paths corresponding to each ID
        
    Returns:
        feature_dict: dict, a dictionary of features for each key'''
    feature_dict = {}
    for class_id, paths in image_paths.items():
        feature_dict[class_id] = [extract_feature(model, transform, path) for path in paths]
    return feature_dict

def avg_features(track_features):
    '''
    Average the features of a track
    
    Args:
        track_features: dict, a dictionary of features for each key
        
    Returns:
        avg_feature_dict: dict, a dictionary of averaged features for each key'''
    avg_feature_dict = {}
    for key, features in track_features.items():
        avg_feature_dict[key] = torch.stack(features).mean(dim=0)
    return avg_feature_dict


def get_class_scores(track_feats, database):
    '''
    Get the similarities between a track and a database of features for each class.
    Returns a list of best score per class.

    Args:
        track_feats: list, a list of feature vectors for the track
        database: dict, a dictionary of features for each class

    Returns:
        scores: dict, a dictionary of best scores for each class
    '''
    scores = {}

    for identity, ref_feats in database.items():
        best_score = -1
        for track_feat in track_feats:
            for ref_feat in ref_feats:
                score = cosine_similarity(
                    track_feat,
                    ref_feat
                )

                if score > best_score:
                    best_score = score
        scores[identity] = best_score

    return scores

def resolve_class_assignments(track_class_dict, class_threshold=0.5):
    '''Resolve the class assignments for each track based on the scores, a threshold and overlapping tracks using while.
    If no score is above the threshold, assign "NoID" class.

    Args:
        track_class_dict: dict, a dictionary containing the scores for each track and class, as well as the indexes of the detections corresponding to each track
        class_threshold: float, the threshold to consider a score as valid for class assignment

    Returns:
        final_assignments: dict, a dictionary containing the final class assignment for each track
    '''
    # Sort scores descending for each track
    ranked_classes = {}
    for track_id, track_info in track_class_dict.items():
        ranked_classes[track_id] = sorted(track_info['scores'].items(), key=lambda x: x[1], reverse=True)

    # Precompute overlaps
    overlaps = defaultdict(set)
    track_ids = list(track_class_dict.keys())

    for i, tid1 in enumerate(track_ids):
        idxs1 = set(track_class_dict[tid1]['idxs'])
        for tid2 in track_ids[i+1:]:
            idxs2 = set(track_class_dict[tid2]['idxs'])
            if idxs1.intersection(idxs2):
                overlaps[tid1].add(tid2)
                overlaps[tid2].add(tid1)

    # Current choice rank for each track
    choice_idx = {tid: 0 for tid in track_ids}

    def get_assignment(track_id):
        ranking = ranked_classes[track_id]
        while choice_idx[track_id] < len(ranking):
            cls, score = ranking[choice_idx[track_id]]
            if score >= class_threshold:
                return cls, score
            break
        return "NoID", 0.0

    changed = True
    while changed:
        changed = False
        current_assignment = {
            tid: get_assignment(tid)
            for tid in track_ids
        }
        for tid1 in track_ids:
            cls1, score1 = current_assignment[tid1]
            if cls1 == "NoID":
                continue
            for tid2 in overlaps[tid1]:
                if tid1 >= tid2:
                    continue
                cls2, score2 = current_assignment[tid2]
                if cls1 != cls2:
                    continue
                # Conflict found
                if score1 >= score2:
                    loser = tid2
                else:
                    loser = tid1
                choice_idx[loser] += 1
                changed = True
                break
            if changed:
                break
    final_assignments = {tid: get_assignment(tid) for tid in track_ids}
    return final_assignments