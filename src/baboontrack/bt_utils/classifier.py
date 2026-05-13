import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

def cosine_similarity(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()

def load_model_and_transform(model_path, device='cpu'):
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


def extract_feature(model, transform, image_path):

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        feat = model(x)

    feat = feat.squeeze()

    # normalize
    feat = feat / feat.norm()

    return feat

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
    for image_path in image_paths:
        feature_dict[image_path] = extract_feature(model, transform, image_path)
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


def classify_track(track_embedding, database):
    best_id = None
    best_score = -1

    for identity, feats in database.items():

        for ref_feat in feats:

            score = cosine_similarity(
                track_embedding,
                ref_feat
            )

            if score > best_score:
                best_score = score
                best_id = identity

    return best_id, best_score