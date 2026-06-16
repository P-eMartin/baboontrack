import torch
import cv2
import numpy as np
from torchvision import transforms
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker

class ReIDModel:
    def __init__(self, device):
        import torchreid
        self.model = torchreid.models.build_model(
            name='osnet_ain_x1_0', # 'osnet_x0_25' 'osnet_ain_x1_0'
            num_classes=20,
            pretrained=True
        )
        self.model.eval()
        self.model.to(device)

        self.transform = self.define_transform()
        self.device = device

    def define_transform(self):
        '''
        Define the transform to apply to the images before feeding them to the re-identification model.
        
        Returns:
            transform: torchvision.transforms, the defined transform
        '''
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract_feature(self, image, bbox, from_BGR=False):
        '''
        Extract the feature of an bbox using the re-identification model.
        
        Args:
            image: np.array, the input image
            bbox: tuple, the bounding box coordinates (x, y, w, h)
            from_BGR: bool, whether the image is in BGR format (default False)
            
        Returns:
            feature: torch.Tensor, the extracted feature of the image
        '''
        x, y, w, h = map(int, bbox)
        crop = image[y:y+h, x:x+w]

        if crop.size == 0:
            return np.zeros(512)
        
        if from_BGR:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(crop).unsqueeze(0)

        with torch.no_grad():
            feature = self.model(tensor)
        feature = feature.cpu().numpy()[0]

        # normalize
        feature /= np.linalg.norm(feature)
        return feature

    def extract_features(self, image, bboxes, from_BGR=False, max_batch_size=10):
        '''
        Extract ReID features for multiple bboxes from a single image.

        Args:
            image: np.ndarray (single frame)
            bboxes: list of (x, y, w, h) of length N
            from_BGR: bool (default False)
            max_batch_size: int

        Returns:
            features: np.ndarray of shape (N, D)
        '''
        crops = []
        valid_indices = []

        # 1. Crop all bboxes
        for i, bbox in enumerate(bboxes):

            x, y, w, h = map(int, bbox)
            crop = image[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            if from_BGR:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops.append(self.transform(crop))
            valid_indices.append(i)

        # handle empty case
        if len(crops) == 0:
            return np.zeros((len(bboxes), 512))

        crops = torch.stack(crops)

        features_list = []

        # 2. Batched inference (chunked)
        with torch.no_grad():

            for i in range(0, len(crops), max_batch_size):

                batch = crops[i:i + max_batch_size].to(self.device)

                out = self.model(batch)

                out = out.cpu().numpy()

                # normalize
                norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
                out = out / norms

                features_list.append(out)

        features = np.vstack(features_list)

        # 3. Restore original order
        final_features = np.zeros((len(bboxes), features.shape[1]))

        for idx, feat in zip(valid_indices, features):
            final_features[idx] = feat

        return final_features
    
    def inference(self, image, bboxes, from_BGR=False, max_batch_size=20):
        '''
        Extract ReID features for multiple bboxes from a single image.

        Args:
            image: np.ndarray (single frame)
            bboxes: list of (x, y, w, h) of length N
            from_BGR: bool (default False)
            max_batch_size: int

        Returns:
            features: np.ndarray of shape (N, D)
        '''
        crops = []
        valid_indices = []

        # 1. Crop all bboxes
        for i, bbox in enumerate(bboxes):

            x, y, w, h = map(int, bbox)
            crop = image[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            if from_BGR:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops.append(self.transform(crop))
            valid_indices.append(i)

        # handle empty case
        if len(crops) == 0:
            return np.zeros((len(bboxes), 512))

        crops = torch.stack(crops)

        features_list = []

        # 2. Batched inference (chunked)
        with torch.no_grad():

            for i in range(0, len(crops), max_batch_size):

                batch = crops[i:i + max_batch_size].to(self.device)

                out = self.model(batch)

                out = out.cpu().numpy()

                # normalize
                norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
                out = out / norms

                features_list.append(out)

        features = np.vstack(features_list)

        # 3. Restore original order
        final_features = np.zeros((len(bboxes), features.shape[1]))

        for idx, feat in zip(valid_indices, features):
            final_features[idx] = feat

        return final_features


def init_tracker(max_cosine_distance=0.5, nn_budget=None, max_iou_distance=0.7, max_age=70, n_init=3):
    '''
    Initialize the tracker.
    
    Args:
        max_cosine_distance: float, the maximum cosine distance for the tracker (default 0.5)
        nn_budget: int, the maximum number of features to store for each track (default None)
        max_iou_distance: float, the maximum IoU distance for the tracker (default 0.7)
        max_age: int, the maximum age of a track (default 70)
        n_init: int, the number of initial detections to initialize a track (default 3)
    Returns:
        tracker: Tracker, the initialized tracker'''
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
    return tracker

def update_tracker(tracker, frame, feat_model, det_list, default_cat_id=1):
    '''
    Update the tracker with a new detection
    
    Args:
        tracker: Tracker, the tracker to update
        frame: np.array, the current video frame
        feat_model: ReIDModel, the feature extraction model with .model and .transform attributes
        det_list: list, a list of detection dictionaries, each containing a 'bbox' key with a bounding box in (x, y, w, h) format in percentage of the frame size
        default_cat_id: int, the default category ID for detections (default 1)
    Returns:
        current_tracks: list, the updated tracks
        max_track_id: int, the maximum track ID
    '''
    image_size = frame.shape[1], frame.shape[0]
    bboxes = []
    scores = []
    # Convert the bounding boxes from percentage to pixel coordinates
    for det in det_list:
        x, y, w, h = det['bbox']
        x = int(x * image_size[0])
        y = int(y * image_size[1])
        w = int(w * image_size[0])
        h = int(h * image_size[1])
        scores.append(det.get('score', 1.0))
        bboxes.append((x, y, w, h))
    features = feat_model.extract_features(frame, bboxes)
    detections = [Detection(bbox, score, feat) for bbox, score, feat in zip(bboxes, scores, features)]
    tracker.predict()
    tracker.update(detections)

    current_tracks = []
    max_track_id = -1
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x, y, w, h = track.to_tlwh()
        track_id = track.track_id
        max_track_id = max(max_track_id, track_id)
        # Back to percentage coordinates
        current_tracks.append({
            'track_id': track.track_id,
            'bbox': [float(x/image_size[0]), float(y/image_size[1]), float(w/image_size[0]), float(h/image_size[1])],
            'score': float(track.score) if hasattr(track, 'score') else 1.0,
            'category_id': default_cat_id
        })
    return current_tracks, max_track_id
