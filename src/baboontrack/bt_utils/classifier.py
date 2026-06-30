from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from .primateface import PrimateFace
from .img_utils import print_and_log, apply_roi_factor

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

def resolve_class_assignments(track_class_dict, class_threshold=0.5, noid_str='NoID'):
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

    def get_assignment(track_id, noid_str):
        ranking = ranked_classes[track_id]
        while choice_idx[track_id] < len(ranking):
            cls, score = ranking[choice_idx[track_id]]
            if score >= class_threshold:
                return cls, score
            break
        return noid_str, 0.0

    changed = True
    while changed:
        changed = False
        current_assignment = {
            tid: get_assignment(tid, noid_str)
            for tid in track_ids
        }
        for tid1 in track_ids:
            cls1, score1 = current_assignment[tid1]
            if cls1 == noid_str:
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
    final_assignments = {tid: get_assignment(tid, noid_str) for tid in track_ids}
    return final_assignments

def cosine_similarity(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()

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

class PrimateFaceDetector:
    def __init__(self, device='cuda', det_thr=0.5, nms_thr=0.4):
        self.pf = PrimateFace(
            device= device,
            pose_model = None,
            det_threshold = det_thr,
            nms_threshold = nms_thr
        )
    
    def detect(self, img):
        bboxes, scores = self.pf._processor.detect_primates(img)
        return bboxes, scores

class MyClassifier:
    def __init__(self, model_path='', device='cpu', detector_type=None, feat_avg=False, nca=False, det_thr=0.5, nms_thr=0.4, epochs=100, lr=1e-4, roi_factor=1.0, log=None):
        self.log = log
        # Load the model
        # model = torch.load(model_path, map_location=device)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Device
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.feat_avg = feat_avg
        self.nca = nca
        self.epochs = epochs
        self.lr = lr
        self.roi_factor = roi_factor

        # Define the transform
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        if not detector_type:
            self.det = None
        elif detector_type == 'primateface':
            self.det = PrimateFaceDetector(
                device=device,
                det_thr=det_thr,
                nms_thr=nms_thr
            )
        else:
            print_and_log(f"Warning: Detector type {detector_type} not recognized. No detector will be used with the classifier.", log=self.log)
            self.det = None

        if nca:
            self.projection = torch.nn.Linear(768, 128)
            self.projection.to(device)
            torch.nn.init.eye_(self.projection.weight[:, :128])
        else:
            self.projection = None
        self.name = 'MyClassifier' \
            + ('_primateface_%g_%g' % (det_thr, nms_thr) if detector_type == 'primateface' else '') \
            + ('_NCA_%d-%g' % (epochs, lr) if nca else '') + ('_featavg' if feat_avg else '') \
            + ('_roi%g' % roi_factor if roi_factor != 1.0 else '')
        
    
    def read_image_cv2(self, img):
        '''
        Read an image from a path or a numpy array with cv2.
        
        Args:
            img: str or numpy.ndarray, the path to the image or the image itself

        Returns:
            numpy.ndarray, the read image in BGR format
        '''
        if isinstance(img, str):
            image = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            image = img
        else:
            raise ValueError("img should be a string (path to the image) or a numpy array (the image itself)")
        return image
    
    def read_image_pil(self, img):
        '''
        Read an image from a path or a numpy array and convert it to RGB if needed.
        
        Args:
            img: str or numpy.ndarray, the path to the image or the image itself

        Returns:
            PIL.Image, the read image
        '''
        if isinstance(img, str):
            image = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            image = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            image = img.convert("RGB")
        else:
            raise ValueError("img should be a string (path to the image) or a numpy array (the image itself)")
        return image

    def train_nca(self, dataloader, epochs=100, lr=1e-4, saved_folder=os.path.join('.tmp', 'nca')):
        # Load the projection layer if it exists
        path_weights = os.path.join(saved_folder, f"{self.name}.pt")
        os.makedirs(saved_folder, exist_ok=True)
        if os.path.exists(path_weights):
            self.projection.load_state_dict(torch.load(path_weights, map_location=self.device))
            print_and_log(f"Loaded NCA projection layer from {path_weights}", log=self.log)
            return
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    feats = self.model(imgs)
                    feats = feats.squeeze()
                    feats = F.normalize(feats, dim=1)
                feats = self.projection(feats)
                loss = self.nca_loss(feats, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}: {total_loss:.4f}")
        # Save nca in .tmp folder to avoid retraining if the same model is used again
        torch.save(self.projection.state_dict(), path_weights)
    
    def nca_loss(self, embeddings, labels, temperature=1.0):
        """
        embeddings: [B, D]
        labels: [B]
        """
        embeddings = F.normalize(embeddings, dim=1)

        dist = torch.cdist(embeddings, embeddings)  # [B,B]
        sim = -dist / temperature

        # mask self
        eye = torch.eye(len(labels), device=labels.device)
        sim = sim - 1e9 * eye

        # softmax over neighbors
        prob = torch.softmax(sim, dim=1)

        labels = labels.unsqueeze(1)
        same = (labels == labels.T).float()

        loss = -torch.log((prob * same).sum(dim=1) + 1e-8).mean()
        return loss

    def project(self, feat):
        if self.projection is None:
            return feat
        feat = self.projection(feat)
        feat = feat / feat.norm()
        return feat

    def extract_feature(self, img):
        '''
        Extract the feature of a BGR image using the model and the transform.
        
        Args:
            img: str or numpy.ndarray, the path to the image or the image itself

        Returns:
            torch.Tensor, the extracted feature
            tuple, the bounding box coordinates (x1, y1, x2, y2)
        '''
        if self.det is not None:
            image = self.read_image_cv2(img)
            if image.size == 0:
                print_and_log("Empty image, cannot extract feature.", log=self.log)
                return None, None
            bboxes, scores = self.det.detect(image)
            if len(bboxes) == 0:
                return None, None
            # Take the bbox with the highest score
            best_idx = np.argmax(scores)
            x1, y1, x2, y2 = apply_roi_factor(bboxes[best_idx], self.roi_factor)
            # max, min and closest integer
            x1, y1, x2, y2 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1))), int(min(image.shape[1], np.ceil(x2))), int(min(image.shape[0], np.ceil(y2)))
            image = self.read_image_pil(image[y1:y2, x1:x2])
            bbox = (x1, y1, x2-x1, y2-y1)
        else:
            image = self.read_image_pil(img)
            bbox = None
        # Apply the transform and add a batch dimension
        x = self.transform(image).unsqueeze(0)
        # Device
        x = x.to(next(self.model.parameters()).device)
        with torch.no_grad():
            feat = self.model(x)
        feat = feat.squeeze()
        # normalize
        feat = feat / feat.norm()
        # Project if needed
        feat = self.project(feat)
        return feat, bbox

    def build_database(self, image_paths):
        '''
        Build a dictionary of features for a list of image paths.
        
        Args:
            image_paths: dict, a dictionary of list of image paths, with the keys being the IDs of the images (e.g. track IDs) and the values being the list of image paths corresponding to each ID
        '''
        if self.nca:
            # Train the NCA projection layer on the features of the images in the database
            from torch.utils.data import Dataset, DataLoader
            class FeatureDataset(Dataset):
                def __init__(self, image_paths, transform):
                    self.image_paths = []
                    self.labels = []
                    self.class_to_idx = {label: idx for idx, label in enumerate(image_paths.keys())}
                    for label, paths in image_paths.items():
                        for path in paths:
                            self.image_paths.append(path)
                            self.labels.append(label)
                    self.transform = transform

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    img_path = self.image_paths[idx]
                    label = self.class_to_idx[self.labels[idx]]
                    image = Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    return image, label
            dataset = FeatureDataset(image_paths, self.transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            self.train_nca(dataloader, epochs=self.epochs, lr=self.lr)
        self.database = {}
        for class_id, paths in image_paths.items():
            id_features = []
            for path in paths:
                feature, _ = self.extract_feature(path)
                if feature is None:
                    print_and_log(f"Little Warning: Could not extract feature from image {path}. But don't worry, other images from the same class will be used.", log=self.log)
                else:
                    id_features.append(feature)
            if len(id_features) == 0:
                print_and_log(f"Warning: No feature could be extracted for class {class_id}. This class cannot be used for classification.", log=self.log)
                self.database[class_id] = None
            elif self.feat_avg:
                avg_feat = torch.stack(id_features).mean(dim=0)
                avg_feat /= avg_feat.norm()
                self.database[class_id] = avg_feat
            else:
                self.database[class_id] = id_features

    def get_class_scores(self, track_feats):
        '''
        Get the similarities between a track and a database of features for each class.
        Returns a list of best score per class.

        Args:
            track_feats: list, a list of feature vectors for the track

        Returns:
            scores: dict, a dictionary of best scores for each class
        '''
        scores = {}
        if self.feat_avg and len(track_feats) > 0:
            avg_track_feat = torch.stack(track_feats).mean(dim=0)
            avg_track_feat /= avg_track_feat.norm()
        for identity, ref_feats in self.database.items():
            if ref_feats is None:
                scores[identity] = -1
                continue
            if self.feat_avg and len(track_feats) > 0:
                scores[identity] = cosine_similarity(avg_track_feat, ref_feats)
            else:
                best_score = -1
                for track_feat in track_feats:
                    for ref_feat in ref_feats:
                        score = cosine_similarity(track_feat, ref_feat)
                        if score > best_score:
                            best_score = score
                scores[identity] = best_score
        return scores