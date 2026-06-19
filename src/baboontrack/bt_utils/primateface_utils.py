import argparse
import os
import torch
from .primateface import PrimateFace
import datetime
import cv2
from io_utils import get_all_files_in_folder

if __name__ == "__main__":
    # Get input arguments - the image or folder or images to process and the output directory
    parser = argparse.ArgumentParser(description="PrimateFace Demo")
    parser.add_argument(
        "--input_path", "-i",
        type=str,
        default='../data/group_dict',
        help="Path to the input image or folder containing images to process.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default='outputs/%s/' % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Directory where the output will be saved.",
    )

    args = parser.parse_args()
    det_thr = 0.5
    nms_thr = 0.1
    
    pf = PrimateFace(
        device= "cuda:0" if torch.cuda.is_available() else "cpu",
        pose_model = None,
        det_threshold = 0
    )
    img_list = get_all_files_in_folder(args.input_path, extensions=['.jpg', '.jpeg', '.png', '.bmp'])
    for img_path in img_list:
        print(f"Processing {img_path}...")

        bgr = cv2.imread(img_path)
        # Run detection
        bboxes, scores = pf._processor.detect_primates(bgr, bbox_thr=det_thr, nms_thr=nms_thr)
        print(f"Detected {len(bboxes)} faces in the image. bboxes: {bboxes}, scores: {scores}")

        # Draw the results on the image and save to output directory
        output_path = os.path.join(args.output_dir, os.path.relpath(img_path, args.input_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, bgr)

        # faces = pf.analyze(img_path)          # detect + 68-point landmarks
        # print(f"Detected {len(faces)} faces in {img_path}.")
        # print(str(faces))
        # # Keep the original file path from the input_path
        # 
        # pf.draw(faces, img_path, output=output_path, draw_keypoints=False, draw_skeleton=False, draw_bbox=True)