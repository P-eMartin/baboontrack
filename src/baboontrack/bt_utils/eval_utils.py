import pdb

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from pathlib import Path
import json
import time
import sys
PYTHON_VERSION = sys.version_info[0]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# To avoid the error "AttributeError: module 'numpy' has no attribute 'asfarray'" when using motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, **kwargs: np.asarray(a, dtype=float)
import csv
import os
import copy
import re
from .io_utils import print_and_log, get_value_with_precision, save_dict_as_csv, zip_folder, get_first_folder
from .json_utils import load_json_file, save_json_file
from collections import defaultdict
import zipfile
from .bot_sort.tracking_utils.evaluation import Evaluator
from .plt_utils import plot_confusion_matrix

class myCOCO(COCO):
    '''
    Modified version of Coco where enmpty detections are handled gracefully instead of raising an error.
    '''
    def loadRes(self, resFile, ignore_idx_mismatch=True, log=None):
        """
        Load result file and return a result api object.

        Args:
            resFile (str or list): filename of result file or the results as a list of dicts.

        Returns:
            myCOCO: result api object
        """
        res = COCO()
        res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {}))
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode): # type: ignore
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        if set(annsImgIds) != (set(annsImgIds) & set(self.getImgIds())):
            if ignore_idx_mismatch:
                missing_img_ids = [i for i in set(annsImgIds) if i not in set(self.getImgIds())]
                print_and_log(f'Warning: The following image IDs are in the results but not in the ground truth: {missing_img_ids}. Adding them to the dataset.', log=log)
                # Add the missing images to the res dataset with empty annotations and using the same resolution as the first image in the ground truth dataset
                resolution = self.dataset['images'][0]['width'], self.dataset['images'][0]['height']
                for img_id in missing_img_ids:
                    self.dataset['images'].append({
                        'id': img_id,
                        'width': resolution[0],
                        'height': resolution[1]
                    })
                # Update the indexes of the ground truth dataset
                self.createIndex()
            else:
                raise Exception('Results do not correspond to current coco set')
            
        # Case when the results are empty
        if len(anns) == 0:
            print('Results is empty.')
            res.dataset['annotations'] = []
            res.createIndex()
            return res
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
    
def bbox_iou(box1, box2):
            """
            COCO bbox format: [x, y, w, h]
            """
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            inter = max(0, xb - xa) * max(0, yb - ya)
            if inter == 0:
                return 0.0
            area1 = w1 * h1
            area2 = w2 * h2

            return inter / (area1 + area2 - inter)

class myCOCOeval(COCOeval):
    '''
    Modified version of Coco eval where classes can be ignored in the evaluation because the annotators are not certain of
    the category of the object. This is useful for example when the annotators are not sure of the identity of the baboon
    in the image and they want to ignore it in the evaluation of classification, but not in the evaluation of detection.
    '''
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', ignore_classes=[]):
        super().__init__(cocoGt, cocoDt, iouType)
        self.ignore_classes = ignore_classes

    def _prepare(self):
        super()._prepare()
        if self.ignore_classes:
            for gt in self._gts:
                for g in self._gts[gt]:
                    if g['category_id'] in self.ignore_classes:
                        g['ignore'] = 1

    def compute_cm(self, iouThr=0.5, remove_unused_classes=True):
        """
        Compute a confusion matrix using IoU matching only (ignoring categories).
        Rows = ground truth categories + FP row (last), columns = predicted categories + FN column (last).

        Ignore key is not taken into account.

        Args:
            iouThr (float): IoU threshold for matching. Default: 0.5.
            remove_unused_classes (bool): If True, remove classes that are not present in the ground truth or predictions. Default: True.

        Returns:
            cm (np.ndarray): Confusion matrix of shape (num_categories + 1, num_categories + 1).
            labels (list): List of category IDs corresponding to the rows and columns of the confusion matrix.
        """

        cat_ids = sorted(self.cocoGt.getCatIds())
        cat_to_idx = {c: i for i, c in enumerate(cat_ids)}

        n = len(cat_ids)

        # last row = FP
        # last col = FN
        cm = np.zeros((n + 1, n + 1), dtype=np.int64)

        for img_id in self.params.imgIds:
            gt = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=[img_id]))
            dt = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=[img_id]))
            if len(gt) == 0 and len(dt) == 0:
                continue
            # highest confidence first
            dt = sorted(dt, key=lambda x: -x["score"])
            matched_gt = set()
            # ------------------------
            # Match detections
            # ------------------------
            for d in dt:
                best_gt = None
                best_iou = iouThr
                for g in gt:
                    if g["id"] in matched_gt:
                        continue
                    if g.get("iscrowd", 0):
                        continue
                    iou = bbox_iou(d["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g

                if best_gt is None:
                    # False positive
                    pred_idx = cat_to_idx[d["category_id"]]
                    cm[-1, pred_idx] += 1
                    continue
                matched_gt.add(best_gt["id"])
                gt_idx = cat_to_idx[best_gt["category_id"]]
                pred_idx = cat_to_idx[d["category_id"]]
                cm[gt_idx, pred_idx] += 1

            # ------------------------
            # False negatives
            # ------------------------
            for g in gt:
                if g["id"] in matched_gt:
                    continue
                # # ignored GT do not count as FN
                # if g.get("ignore", 0):
                #     continue
                gt_idx = cat_to_idx[g["category_id"]]
                cm[gt_idx, -1] += 1
        labels = cat_ids + ["FP/FN"]
        if remove_unused_classes:
            # Ignore the FP row and FN column
            row_sum = cm[:-1, :].sum(axis=1)   # GT occurrences
            col_sum = cm[:, :-1].sum(axis=0)   # Prediction occurrences

            keep = (row_sum + col_sum) > 0

            cm = np.block([
                [cm[:-1, :-1][keep][:, keep], cm[:-1, -1][keep, None]],
                [cm[-1, :-1][None, keep], cm[-1:, -1:]]
            ])

            labels = [l for l, k in zip(cat_ids, keep) if k] + ["FP/FN"]
        return cm, labels
    
    def plot_pr_curve(self, cat_id=None, iouThr=0.5, area="all", maxDet=None, save_path=None):
        """
        Plot COCO Precision-Recall curves.

        Args:
            cat_id (int or None):
                Category ID to plot. If None, plots all categories.
            iouThr (float):
                IoU threshold to display (default=0.5).
            area (str):
                One of self.params.areaRngLbl (default="all").
            maxDet (int or None):
                maxDet value to use. Defaults to largest available.
            save_path (str or None):
                If provided, saves the figure instead of displaying it.
        """
        if self.eval is None or "precision" not in self.eval:
            raise RuntimeError(
                "Call evaluate() and accumulate() before plotting PR curves."
            )

        precision = self.eval["precision"]
        recall = self.params.recThrs

        # IoU index
        try:
            t = np.where(np.isclose(self.params.iouThrs, iouThr))[0][0]
        except IndexError:
            raise ValueError(f"IoU={iouThr} not found. Available: {self.params.iouThrs}")

        # Area index
        try:
            a = self.params.areaRngLbl.index(area)
        except ValueError:
            raise ValueError(f"Unknown area '{area}'. Choices: {self.params.areaRngLbl}")

        # maxDet index
        if maxDet is None:
            m = len(self.params.maxDets) - 1
            maxDet = self.params.maxDets[m]
        else:
            try:
                m = self.params.maxDets.index(maxDet)
            except ValueError:
                raise ValueError(f"maxDet={maxDet} not found. Choices: {self.params.maxDets}")

        plt.figure(figsize=(6, 6))

        if cat_id is None:
            cat_ids = [
                cid for cid in self.params.catIds
                if cid not in self.ignore_classes
            ]
        else:
            if cat_id not in self.params.catIds:
                raise ValueError(f"Category {cat_id} not evaluated.")
            if cat_id in self.ignore_classes:
                raise ValueError(
                    f"Category {cat_id} is ignored and should not be plotted."
                )
            cat_ids = [cat_id]

        for cid in cat_ids:
            k = self.params.catIds.index(cid)
            p = precision[t, :, k, a, m]
            valid = p > -1
            if not np.any(valid):
                continue
            ap = np.mean(p[valid])

            # Use category name if available
            try:
                name = self.cocoGt.loadCats([cid])[0]["name"]
            except Exception:
                name = str(cid)
            plt.plot(recall[valid], p[valid], lw=2, label=f"{name} (AP={ap:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve @ IoU={iouThr:.2f}")
        plt.xlim(0, 1)
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def export_best_worst_classifications(self, save_dir, iouThr=0.5, n=50, unique_reference=True):
        """
        Export HTML reports of classification performance.

        Generates:
            - best.html
            - worst.html
            - reference_quality.html

        Parameters
        ----------
        save_dir : str
            Output folder.

        n : int
            Number of samples for best/worst reports.

        unique_reference : bool
            If True, keep only one sample per reference image for best/worst.
        """
        os.makedirs(save_dir, exist_ok=True)
        symlink_created = False
        id_to_name = {
            cid: cat["name"].lower()
            for cid, cat in self.cocoGt.cats.items()
        }
        name_to_id = {
            cat["name"].lower(): cid
            for cid, cat in self.cocoGt.cats.items()
        }
        # ----------------------------------------------------
        # Recover GT labels from IoU matching
        # ----------------------------------------------------
        gt_lookup = {}
        for img_id in self.params.imgIds:
            gt = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=[img_id]))
            dt = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=[img_id]))
            matched = set()
            for d in sorted(dt, key=lambda x: -x["score"]):
                best = None
                best_iou = iouThr
                for g in gt:
                    if g["id"] in matched:
                        continue
                    iou = bbox_iou(d["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best = g
                if best is None:
                    continue
                matched.add(best["id"])
                gt_lookup[d["id"]] = best["category_id"]

        # ----------------------------------------------------
        # Build samples
        # ----------------------------------------------------
        samples = []
        for ann in self.cocoDt.dataset["annotations"]:
            if ann["id"] not in gt_lookup:
                continue
            # Check if the annotation has the required attributes
            if "path_ref" not in ann['attributes'] or "path_crop" not in ann['attributes']:
                continue

            gt_id = gt_lookup[ann["id"]]
            pred_id = ann["category_id"]
            gt_name = id_to_name[gt_id]
            pred_name = id_to_name[pred_id]
            pred_score = float(ann["score"])
            class_scores = ann["attributes"].get("class_scores", {})
            # Lowercase the class names in class_scores for consistency and sort them by score in descending order
            class_scores = {name.lower(): (_score, _crop_path, _ref_path) for name, (_score, _crop_path, _ref_path) in class_scores.items()}
            # Get the highest score
            highest_score = max((score for score, _, _ in class_scores.values()), default=0.0)

            # Check if the attributes are not None
            ref_path = ann['attributes']["path_ref"]
            crop_path = ann['attributes']["path_crop"]

            # ref and crop paths processing
            if ref_path is None or crop_path is None:
                continue
            if not symlink_created:
                # Remove existing symlinks if they exist
                ref_symlink = os.path.join(save_dir, "ref")
                crop_symlink = os.path.join(save_dir, "crop")
                if os.path.islink(ref_symlink):
                    os.remove(ref_symlink)
                if os.path.islink(crop_symlink):
                    os.remove(crop_symlink)
                # Wait for a moment to ensure the filesystem has updated
                time.sleep(0.1)
                # Create symlinks to ref and crop root folders in the save_dir. Works only if the ref and crop first folders are the same for all samples
                os.symlink(os.path.relpath(get_first_folder(ref_path), save_dir), ref_symlink)
                os.symlink(os.path.relpath(get_first_folder(crop_path), save_dir), crop_symlink)
                symlink_created = True

            # Determine the GT score and paths based on the prediction
            if gt_name == pred_name:
                # the winning class
                gt_score = pred_score
                gt_crop = crop_path
                gt_ref = ref_path
                margin = gt_score - highest_score
            elif gt_name in class_scores:
                gt_score, gt_crop, gt_ref = class_scores[gt_name]
                margin = gt_score - pred_score
            else:
                gt_score = 0.0
                gt_crop = None
                gt_ref = None
            
            samples.append(
                {
                    # GT Prediction
                    "gt": gt_id,
                    "gt_name": gt_name,
                    "gt_score": gt_score,
                    "gt_roi": os.path.join('crop', os.path.relpath(gt_crop, get_first_folder(gt_crop)).lstrip(os.sep)) if gt_crop else None,
                    "gt_ref": os.path.join('ref', os.path.relpath(gt_ref, get_first_folder(gt_ref)).lstrip(os.sep)) if gt_ref else None,

                    # Best Prediction
                    "pred": pred_id,
                    "pred_name": pred_name,
                    "pred_score": pred_score,
                    "pred_roi": os.path.join('crop', os.path.relpath(crop_path, get_first_folder(crop_path)).lstrip(os.sep)),
                    "pred_ref": os.path.join('ref', os.path.relpath(ref_path, get_first_folder(ref_path)).lstrip(os.sep)),

                    # Other
                    "track": ann['attributes'].get("track_id"),
                    'extra_bbox': ann['attributes'].get("extra_bbox"),
                    "image": ann["image_id"],
                    "margin": margin,
                }
            )
        if len(samples) == 0:
            print("No samples available for classification report.")
            return

        # ----------------------------------------------------
        # Reference image statistics
        # ----------------------------------------------------
        ref_stats = defaultdict(
            lambda: {
                "correct": 0,
                "wrong": 0,
                "scores": [],
                "errors": [],
                "tracks": {},
            }
        )
        for s in samples:
            stat = ref_stats[s["pred_ref"]]
            stat["scores"].append(s["pred_score"])
            if s["gt"] == s["pred"]:
                stat["correct"] += 1
            else:
                stat["wrong"] += 1
                stat["errors"].append(s)
            # Track-level storage
            track_id = s.get("track")
            if track_id is not None:
                if track_id not in stat["tracks"]:
                    stat["tracks"][track_id] = {
                        "correct": 0,
                        "wrong": 0,
                        "scores": [],
                        "samples": [],
                    }
                track = stat["tracks"][track_id]
                track["scores"].append(s["pred_score"])
                track["samples"].append(s)
                if s["gt"] == s["pred"]:
                    track["correct"] += 1
                else:
                    track["wrong"] += 1
        reference_quality = []
        for ref, stat in ref_stats.items():
            total = stat["correct"] + stat["wrong"]
            track_results = []
            for track_id, track in stat["tracks"].items():
                n = track["correct"] + track["wrong"]
                track_results.append(
                    {
                        "track_id": track_id,
                        "accuracy": track["correct"] / max(n, 1),
                        "correct": track["correct"],
                        "wrong": track["wrong"],
                        "mean_score": np.mean(track["scores"]),
                    }
                )
            correct_tracks = sum(
                t["accuracy"] == 1.0 for t in track_results
            )
            wrong_tracks = sum(
                t["accuracy"] < 1.0 for t in track_results
            )
            reference_quality.append(
                {
                    "ref": ref,
                    "used": total,
                    # detection-level
                    "correct": stat["correct"],
                    "wrong": stat["wrong"],
                    "accuracy": stat["correct"] / max(total, 1),
                    # track-level
                    "tracks": len(track_results),
                    "correct_tracks": correct_tracks,
                    "wrong_tracks": wrong_tracks,
                    "track_accuracy": (
                        correct_tracks / max(len(track_results), 1)
                    ),
                    "mean_score": sum(stat["scores"]) / len(stat["scores"]),
                }
            )
        # Worst references first
        reference_quality = sorted(
            reference_quality,
            key=lambda x: (
                x["accuracy"],
                -x["used"]
            )
        )
        self._write_reference_html(
            os.path.join(save_dir, "reference_quality.html"),
            reference_quality[:n],
        )

        # ----------------------------------------------------
        # Best / worst individual samples
        # ----------------------------------------------------
        gallery_samples = samples
        if unique_reference:
            unique = {}
            for s in gallery_samples:
                ref = s["pred_ref"]
                if ref not in unique or s["pred_score"] > unique[ref]["pred_score"]:
                    unique[ref] = s
            gallery_samples = list(unique.values())
        # Correct high-confidence predictions
        best_samples_score = sorted([s for s in gallery_samples if s["gt"] == s["pred"]], key=lambda x: -x["pred_score"])[:n]
        best_samples_margin = sorted([s for s in gallery_samples if s["gt"] == s["pred"]], key=lambda x: -x["margin"])[:n]
        # Wrong high-confidence predictions
        worst_samples_score_pred = sorted([s for s in gallery_samples if s["gt"] != s["pred"]], key=lambda x: -x["pred_score"])[:n]
        worst_samples_score_gt = sorted([s for s in gallery_samples if s["gt"] != s["pred"]], key=lambda x: x["gt_score"])[:n]
        worst_samples_margin = sorted([s for s in gallery_samples if s["gt"] != s["pred"]], key=lambda x: x["margin"])[:n]
        self._write_html(
            os.path.join(save_dir, "all.html"),
            gallery_samples,
            title="All classifications",
        )
        self._write_html(
            os.path.join(save_dir, "all_margin_sorted.html"),
            sorted(gallery_samples, key=lambda x: -x["margin"]),
            title="All classifications",
        )
        self._write_html_correct(
            os.path.join(save_dir, "best_score.html"),
            best_samples_score,
            title="Best classifications wrt score",
        )
        self._write_html_correct(
            os.path.join(save_dir, "best_margin.html"),
            best_samples_margin,
            title="Best classifications wrt margin",
        )
        self._write_html(
            os.path.join(save_dir, "worst_pred.html"),
            worst_samples_score_pred,
            title="Worst classifications wrt prediction score",
        )
        self._write_html(
            os.path.join(save_dir, "worst_gt.html"),
            worst_samples_score_gt,
            title="Worst classifications wrt GT score",
        )
        self._write_html(
            os.path.join(save_dir, "worst_margin.html"),
            worst_samples_margin,
            title="Worst classifications wrt margin",
        )

    def _write_html(self, filename, samples, title):
        html = [
            "<html>",
            """
            <style>
            table {
                border-collapse: collapse;
            }
            th, td {
                text-align: center;
                vertical-align: middle;
                padding: 6px;
            }
            th {
                background-color: #f0f0f0;
            }
            img {
                border-radius: 4px;
            }
            .correct {
                background-color: #eaf8ea;
            }
            .wrong {
                background-color: #fdeaea;
            }
            </style>
            """,
            "<head>",
            f"<title>{title}</title>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            "<table border='1' cellspacing='0' cellpadding='5'>",
            # First header row
            """
            <tr>
                <th rowspan="2">Margin</th>
                <th colspan="4">Ground Truth</th>
                <th colspan="4">Prediction</th>
            </tr>
            """,
            # Second header row
            """
            <tr>
                <th>Identity</th>
                <th>Score</th>
                <th>ROI</th>
                <th>Reference</th>

                <th>Identity</th>
                <th>Score</th>
                <th>ROI</th>
                <th>Reference</th>
            </tr>
            """,
        ]

        for s in samples:
            row_class = "correct" if s["gt"] == s["pred"] else "wrong"
            html.append(
                f"""
                <tr class="{row_class}">
                <td>{s['margin']:.3f}</td>

                <td>{s['gt']}:{s['gt_name']}</td>
                <td>{s['gt_score']:.3f}</td>
                <td><a href="{s['gt_roi']}"><img src="{s['gt_roi']}" width="180"></a></td>
                <td><a href="{s['gt_ref']}"><img src="{s['gt_ref']}" width="180"></a></td>

                <td>{s['pred']}:{s['pred_name']}</td>
                <td>{s['pred_score']:.3f}</td>
                <td><a href="{s['pred_roi']}"><img src="{s['pred_roi']}" width="180"></a></td>
                <td><a href="{s['pred_ref']}"><img src="{s['pred_ref']}" width="180"></a></td>

                </tr>
                """
            )

        html.extend(["</table>", "</body>", "</html>"])

        with open(filename, "w") as f:
            f.write("\n".join(html))

    def _write_html_correct(self, filename, samples, title):
        html = [
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            "<table border='1' cellspacing='0' cellpadding='5'>",
            "<tr>"
            "<th>margin</th>"
            "<th>Identity</th>"
            "<th>Score</th>"
            "<th>ROI</th>"
            "<th>Reference</th>"
            "</tr>",
        ]

        for s in samples:

            html.append(
                f"""
                <tr>
                <td>{s['margin']:.3f}</td>
                <td>{s['gt']}:{s['gt_name']}</td>
                <td>{s['gt_score']:.3f}</td>
                <td><a href="{s['gt_roi']}">
                <img src="{s['gt_roi']}" width="180">
                </a></td>
                <td><a href="{s['gt_ref']}">
                <img src="{s['gt_ref']}" width="180">
                </a></td>
                </tr>
                """
            )

        html.extend(["</table>", "</body>", "</html>"])

        with open(filename, "w") as f:
            f.write("\n".join(html))

    def _write_reference_html(self, filename, references):
        html = [
            "<html>",
            "<body>",
            "<h1>Reference image quality</h1>",
            "<table border='1' cellspacing='0' cellpadding='5'>",
            """
            <tr>
            <th>Reference</th>

            <th>Frames</th>
            <th>Correct frames</th>
            <th>Wrong frames</th>
            <th>Frame accuracy</th>

            <th>Tracks</th>
            <th>Correct tracks</th>
            <th>Wrong tracks</th>
            <th>Track accuracy</th>

            <th>Mean score</th>
            </tr>
            """
        ]

        for r in references:

            html.append(
            f"""
            <tr>

            <td>
            <a href="{r['ref']}">
            <img src="{r['ref']}" width="250">
            </a>
            </td>

            <td>{r['used']}</td>
            <td>{r['correct']}</td>
            <td>{r['wrong']}</td>
            <td>{100*r['accuracy']:.1f}%</td>


            <td>{r['tracks']}</td>
            <td>{r['correct_tracks']}</td>
            <td>{r['wrong_tracks']}</td>
            <td>{100*r['track_accuracy']:.1f}%</td>


            <td>{r['mean_score']:.3f}</td>

            </tr>
            """
            )

        html.extend(
            [
                "</table>",
                "</body>",
                "</html>",
            ]
        )

        with open(filename, "w") as f:
            f.write("\n".join(html))

'''
Helper
'''
def extract_boundaries(filename, log=None):
    """
    Extract frame boundaries from filenames like:
    - frame-1-546-1639-2000-mot
    - frame-1-546-mot.zip
    - frames.zip

    Returns:
        list of (start, end) tuples or None
    """

    base = os.path.basename(filename)
    base = base.replace(".zip", "")

    # Extract all integers in order
    nums = list(map(int, re.findall(r"\d+", base)))

    # If no numbers → no boundaries
    if not nums:
        return None

    # Must be even number of values to form pairs
    if len(nums) % 2 != 0:
        print_and_log(f"Odd number of boundary values found in: {filename}. Setting to None.", log=log)
        return None

    # Pair them
    boundaries = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    print_and_log(f"Extracted boundaries from {filename}: {boundaries}", log=log)

    return boundaries

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
            'score': det.get('score'),
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
    for idx, frame_dets in enumerate(perso_list):
        for det in frame_dets:
            det['image_id'] = idx + 1
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
        boundaries: list of tuple of int, the boundaries of the frames to save (default None)
        frame_id_offset: int, the offset to add to the frame IDs (default 0, no offset)

    Returns:
        int, 1 if the file was properly saved
    '''
    # Create the COCO format dictionary
    coco_list = []
    # Loop over the detection_dict and fill the COCO format dictionary
    for idx, dets_or_key in enumerate(detection_dict):
        if isinstance(dets_or_key, list): # Case when dets_or_key is a list of detections
            if boundaries and not (any(start <= idx <= end for start, end in boundaries)):
                continue
            for det in dets_or_key:
                coco_list.append({
                    'image_id': idx + 1 + frame_id_offset,
                    'category_id': det['category_id'],
                    'bbox': [
                        get_value_with_precision(det['bbox'][0] * image_size[0] if image_size is not None else det['bbox'][0], 10),
                        get_value_with_precision(det['bbox'][1] * image_size[1] if image_size is not None else det['bbox'][1], 10),
                        get_value_with_precision(det['bbox'][2] * image_size[0] if image_size is not None else det['bbox'][2], 10),
                        get_value_with_precision(det['bbox'][3] * image_size[1] if image_size is not None else det['bbox'][3], 10)
                    ],
                    'score': get_value_with_precision(det.get('score')),
                    'attributes': {
                        'name': 'NoID',
                        'track_id': int(det['track_id'] + 1),
                        'visibility': det.get('visibility', 1),
                        'path_ref': det.get('path_ref', None),
                        'path_crop': det.get('path_crop', None),
                        'class_scores': det.get('class_scores', None),
                        'extra_bbox': det.get('extra_bbox', None)
                    }
                })
        else: # Case when detection_dict is a list of dictionaries (already in COCO format)
            det = dets_or_key
            if boundaries and not (any(start <= det['image_id']-1 <= end for start, end in boundaries)):
                continue
            coco_list.append({
                'image_id': det['image_id'] + frame_id_offset,
                'category_id': det['category_id'],
                'bbox': [
                    get_value_with_precision(det['bbox'][0] * image_size[0], 10) if image_size is not None else det['bbox'][0],
                    get_value_with_precision(det['bbox'][1] * image_size[1], 10) if image_size is not None else det['bbox'][1],
                    get_value_with_precision(det['bbox'][2] * image_size[0], 10) if image_size is not None else det['bbox'][2],
                    get_value_with_precision(det['bbox'][3] * image_size[1], 10) if image_size is not None else det['bbox'][3]
                ],
                'score': get_value_with_precision(det.get('score')),
                'area': det['area'] if 'area' in det else get_value_with_precision(det['bbox'][2] * det['bbox'][3] if image_size is None else det['bbox'][2] * image_size[0] * det['bbox'][3] * image_size[1], 10),
                'iscrowd': det['iscrowd'] if 'iscrowd' in det else 0,
                'attributes': {
                    'name': 'NoID',
                    'track_id': det.get('track_id', det.get('attributes', {}).get('track_id')),
                    'visibility': det['visibility'] if 'visibility' in det else 1,
                    'path_ref': det.get('path_ref', None),
                    'path_crop': det.get('path_crop', None),
                    'class_scores': det.get('class_scores', None),
                    'extra_bbox': det.get('extra_bbox', None)
                }
            })
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'detections.json')
    save_json_file(coco_list, output_file)
    save_json_file(coco_list, output_file.replace('.json', '.pretty.json'), pretty=True)
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
        boundaries: list of tuple of int, the boundaries of the frames to save (default None)
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
            if boundaries and not (any(start <= idx <= end for start, end in boundaries)):
                continue
            for det in dets_or_key:
                mot_dict['frame_id'].append(idx+frame_id_offset)
                mot_dict['track_id'].append(det['track_id']+1)
                mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
                mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
                mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
                mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
                mot_dict['not ignored'].append(1)
                mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else det['category_id'])
                mot_dict['visibility'].append(det.get('visibility', 1))
                mot_dict['skipped'].append(0)
        elif isinstance(detection_dict, dict): # Case when detection_dict is a dictionary with dets_or_key being the key.
            if boundaries and not (any(start <= int(dets_or_key)-1 <= end for start, end in boundaries)):
                continue
            dets = detection_dict[dets_or_key]
            for det in dets:
                if isinstance(det, dict):
                    mot_dict['frame_id'].append(dets_or_key + frame_id_offset -1)
                    mot_dict['track_id'].append(det['track_id'])
                    mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
                    mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
                    mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
                    mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
                    mot_dict['not ignored'].append(det.get('not_ignored', 1))
                    mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else det['category_id'])
                    mot_dict['visibility'].append(det.get('visibility', 1))
                    mot_dict['skipped'].append(det.get('skipped', 0))
                elif isinstance(det, list):
                    for _det in det:
                        mot_dict['frame_id'].append(dets_or_key + frame_id_offset -1)
                        mot_dict['track_id'].append(_det['track_id'])
                        mot_dict['x'].append(int(_det['bbox'][0] * image_size[0]) if image_size is not None else _det['bbox'][0])
                        mot_dict['y'].append(int(_det['bbox'][1] * image_size[1]) if image_size is not None else _det['bbox'][1])
                        mot_dict['w'].append(int(_det['bbox'][2] * image_size[0]) if image_size is not None else _det['bbox'][2])
                        mot_dict['h'].append(int(_det['bbox'][3] * image_size[1]) if image_size is not None else _det['bbox'][3])
                        mot_dict['not ignored'].append(_det.get('not_ignored', 1))
                        mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else _det['category_id'])
                        mot_dict['visibility'].append(_det.get('visibility', 1))
                        mot_dict['skipped'].append(_det.get('skipped', 0))
        else: # Case when detection_dict is a list of dictionaries (in COCO format)
            det = dets_or_key
            if boundaries and not (any(start <= det['image_id']-1 <= end for start, end in boundaries)):
                continue
            mot_dict['frame_id'].append(det['image_id'] + frame_id_offset -1)
            mot_dict['track_id'].append(det['track_id'])
            mot_dict['x'].append(int(det['bbox'][0] * image_size[0]) if image_size is not None else det['bbox'][0])
            mot_dict['y'].append(int(det['bbox'][1] * image_size[1]) if image_size is not None else det['bbox'][1])
            mot_dict['w'].append(int(det['bbox'][2] * image_size[0]) if image_size is not None else det['bbox'][2])
            mot_dict['h'].append(int(det['bbox'][3] * image_size[1]) if image_size is not None else det['bbox'][3])
            mot_dict['not ignored'].append(det.get('not_ignored', 1))
            mot_dict['class_id'].append(cat_id_override if cat_id_override is not None else det['category_id'])
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
        boundaries: list of tuple of int, the start and end frame IDs to load (default None, if None, all frames are loaded)
        log: logging.Logger, the logger to log the information (default None)

    Returns:
        dict: a dictionary containing the loaded detection/tracking results, with frame IDs as keys and
            lists of detections as values. Each detection is a dictionary with keys 'id','track_id', 'bbox', 'category_id', 'visibility', 'not_ignored', and 'skipped'.
    '''
    labels = None
    def parse_rows(reader):
        detection_dict = defaultdict(list)

        for idx, row in enumerate(reader):
            frame_id = int(row[0])

            detection_dict[frame_id].append({
                "id": idx+1,
                "track_id": int(row[1]),
                "bbox": [
                    max(int(float(row[2])), 0),
                    max(int(float(row[3])), 0),
                    max(int(float(row[4])), 0),
                    max(int(float(row[5])), 0)
                ],
                "category_id": int(float(row[7])),
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
            with zf.open("gt/labels.txt") as f:
                labels = [line.decode().strip() for line in f]

    elif os.path.isdir(input_file):
        with open(os.path.join(input_file, "gt", "gt.txt"), newline="") as f:
            detection_dict = parse_rows(csv.reader(f))
        with open(os.path.join(input_file, "gt", "labels.txt"), "r") as f:
            labels = [line.strip() for line in f]

    elif os.path.isfile(input_file):
        with open(input_file, newline="") as f:
            detection_dict = parse_rows(csv.reader(f))

    else:
        raise ValueError(
            f"Input '{input_file}' is not a valid zip file, folder, or file."
        )

    if boundaries is not None:
        detection_dict = {
            frame_id: dets
            for frame_id, dets in detection_dict.items()
            if any(start <= frame_id-1 <= end for start, end in boundaries)
        }

    return detection_dict, labels

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
                'category_id': cat_id_override if cat_id_override is not None else det['category_id'],
                'bbox': [
                    get_value_with_precision(det['bbox'][0] * image_size[0] if image_size is not None else det['bbox'][0], 10),
                    get_value_with_precision(det['bbox'][1] * image_size[1] if image_size is not None else det['bbox'][1], 10),
                    get_value_with_precision(det['bbox'][2] * image_size[0] if image_size is not None else det['bbox'][2], 10),
                    get_value_with_precision(det['bbox'][3] * image_size[1] if image_size is not None else det['bbox'][3], 10)
                ],
                'score': get_value_with_precision(det.get('score')),
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
        mot_dict, mot_categories = load_mot_format(mot_gt_file)
        if categories is None:
            categories = mot_categories
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

def solve_id_conflicts(_detections, labels_input, labels_output, default_label="NoID", log=None):
    '''
    Solve ID conflicts between input labels (name) and expetected labels by matching labels from input
    to the expected labels. Check if name match (lower case) and modify id accordingly. If no match, assign default label and log the conflict.

    Args:
        detections: list of dict, the list of detections with 'category_id' key
        labels_input: list of str, the list of input labels (names)
        labels_output: list of str, the list of expected output labels (names)
        default_label: str, the default label to assign in case of conflict (default "NoID")
        log: logging.Logger, the logger to log the information (default None)
    
    Returns:
        list of dict, the list of detections with resolved ID conflicts
    '''
    detections = copy.deepcopy(_detections) # To avoid modifying the original detections
    name_to_id_output = {label.lower(): idx+1 for idx, label in enumerate(labels_output)}
    nb_id_conflict = 0
    total_detections = 0
    for det in detections:
        if isinstance(det, dict):
            total_detections += 1
            id_input = det['category_id']
            name = labels_input[id_input-1].lower()
            if name in name_to_id_output:
                det['category_id'] = name_to_id_output[name]
            else:
                nb_id_conflict += 1
                det['category_id'] = name_to_id_output.get(default_label.lower(), 0)
        elif isinstance(det, list):
            for _det in det:
                total_detections += 1
                id_input = _det['category_id']
                name = labels_input[id_input-1].lower()
                if name in name_to_id_output:
                    _det['category_id'] = name_to_id_output[name]
                else:
                    nb_id_conflict += 1
                    _det['category_id'] = name_to_id_output.get(default_label.lower(), 0)
    print_and_log(f"Total ID conflicts resolved: {nb_id_conflict} over {total_detections} detections.", log=log)
    return detections

#####
## Evaluation of the detection and tracking results
#####
def coco_eval(gt_file, detection_file, ignore_classes=[], cm=False, pr='', visu='', log=None):
    """
    Run COCO evaluation and return metrics as a flat dictionary.

    Args:
        gt_file: str, path to the ground truth JSON file or zip file in COCO format
        detection_file: str, path to the detection JSON file or zip file in COCO format
        ignore_classes: list of int, list of category IDs to ignore in the evaluation (default empty list)
        cm: bool, whether to compute the confusion matrix (default False)
        pr: str, path to save the precision-recall curve (default empty string)
        visu: str, path to save the visualization of best/worst detections (default empty string)
        log: logging.Logger, the logger to log the information (default None)

    Returns:
        dict, the evaluation results
    """

    coco_gt = myCOCO(gt_file)
    # Check quickly if there are any detections otherwise loading lead to error:
    # if len(load_json_file(detection_file)) == 0:
    #     coco_dt = coco_gt.loadRes([{'image_id':1,'id':0,'category_id':1,'bbox':[0,0,0,0],'score':0}])
    # else:
    coco_dt = coco_gt.loadRes(detection_file, log=log)
    
    my_eval = myCOCOeval(coco_gt, coco_dt, iouType="bbox", ignore_classes=ignore_classes)
    my_eval.evaluate()
    my_eval.accumulate()
    my_eval.summarize()

    metrics = {
        "AP": my_eval.stats[0],
        "AP50": my_eval.stats[1],
        "AP75": my_eval.stats[2],
        "AP_small": my_eval.stats[3],
        "AP_medium": my_eval.stats[4],
        "AP_large": my_eval.stats[5],
        "AR": my_eval.stats[6],
        "AR50": my_eval.stats[7],
        "AR75": my_eval.stats[8],
        "AR_small": my_eval.stats[9],
        "AR_medium": my_eval.stats[10],
        "AR_large": my_eval.stats[11],
    }
    if cm:
        metrics["cm"] = my_eval.compute_cm()
    if pr:
        my_eval.plot_pr_curve(save_path=pr)
    if visu:
        my_eval.export_best_worst_classifications(save_dir=visu)
    return metrics


def evaluate_detection(gt_file, detection_file, name=None, save_path=None, extra_info=None, ignore_classes=[], cm=False, pr='', visu='', log=None):
    '''
    Evaluate the detection performance using COCO metrics.

    Args:
        gt_file: str, path to the ground truth JSON file or zip file in COCO format
        detection_file: str, path to the detection JSON file or zip file in COCO format
        save_path: str, path to save the evaluation results (default None)
        extra_info: dict (optional), additional metadata to include in the evaluation results
        ignore_classes: list of int, list of category IDs to ignore in the evaluation (default empty list)
        cm: bool, whether to compute the confusion matrix (default False)
        pr: str, path to save the precision-recall curve (default empty string)
        visu: path, path to save the visualization of best/worst detections (default empty string)
        log: logging.Logger, the logger to log the information (default None)

    Returns:
        dict, the evaluation results
    '''
    metrics = coco_eval(gt_file, detection_file, ignore_classes=ignore_classes, cm=cm, pr=pr, visu=visu, log=log)
    if save_path is not None:
        update_eval_csv(
            csv_path=save_path,
            segment_name=name if name is not None else os.path.basename(detection_file).split('.')[0],
            metrics={k: v for k, v in metrics.items() if k != "cm"},
            extra_info=extra_info
        )
    return metrics

def mot_eval(gt_file, tracking_file):
    """
    Evaluate MOT tracking performance and return a flat dictionary.
    Can process file/file or folder/folder (in which case it will return the average metrics across all files).

    Args:
        gt_file: str, path to the ground truth MOT file (can be a zip file, a folder containing gt/gt.txt, or a gt.txt file)
        tracking_file: str, path to the tracking results MOT file (can be a zip file, a folder containing gt/gt.txt, or a gt.txt file)

    Returns:
        dict, the evaluation results
    """
    if isinstance(gt_file, list) and isinstance(tracking_file, list):
        print("Warning: this works only if track_id are updated properly across sequences (not overlapping ids) unless the files are meant to be evaluated together. If not, the evaluation will be incorrect.")
        # If both are lists, evaluate each pair of files and average the results
        if len(gt_file) != len(tracking_file):
            print("Length of gt_file and tracking_file lists are not the same. Skipping evaluation.")
            return {}
        elif len(gt_file) == 0:
            print("gt_file and tracking_file lists are empty. Skipping evaluation.")
            return {}
        for idx, (gt_f, tr_f) in enumerate(zip(gt_file, tracking_file)):
            data_root = os.path.dirname(gt_f)
            seq_name = os.path.basename(gt_f).split('.')[0]
            if idx == 0:
                # Initialize the evaluator with the first sequence
                evaluator = Evaluator(data_root, seq_name, data_type="mot")
            else:
                # Update the evaluator with the next sequence
                evaluator.data_root = data_root
                evaluator.seq_name = seq_name
                evaluator.load_annotations()
            eval_results = evaluator.eval_file(os.path.join(tr_f, 'gt', 'gt.txt') if os.path.isdir(tr_f) else tr_f, reset_accumulator=False)
    else:
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

def merge_mot_formats(gt_files, pred_mot_files, output_folder):
    '''
    Merge multiple MOT-format files into a single MOT-format file to be able to evaluate
    a whole dataset at once. The merged file will be saved in the output_folder.
    The track_id and frame_id will be updated accordingly to avoid conflicts.
    We need both gt_files and pred_mot_files in order to be able to modify properly the track_id and frame_id in the merged file.

    Args:
        gt_files: list of str, paths to the ground truth MOT-format files (can be zip files, folders containing gt/gt.txt, or gt.txt files)
        pred_mot_files: list of str, paths to the prediction MOT-format files (can be zip files, folders containing pred/pred.txt, or pred.txt files)
        output_folder: str, path to the folder where the merged file will be saved

    Returns:
        list of str, paths to the merged MOT-format files
    '''
    if len(gt_files) != len(pred_mot_files):
        raise ValueError("The number of ground truth files and prediction files must be the same.")
    merged_gt = defaultdict(list)
    merged_pred = defaultdict(list)
    track_id_offset_gt = 0
    track_id_offset_pred = 0
    frame_id_offset = 0
    for gt_file, pred_file in zip(gt_files, pred_mot_files):
        gt_dict, _ = load_mot_format(gt_file)
        pred_dict, _ = load_mot_format(pred_file)
        # Update track_id and frame_id in gt_dict
        if frame_id_offset > 0 and track_id_offset_gt > 0:
            for frame_id, dets in gt_dict.items():
                for det in dets:
                    det['track_id'] += track_id_offset_gt
        # Update track_id and frame_id in pred_dict
        if frame_id_offset > 0 and track_id_offset_pred > 0:
            for frame_id, dets in pred_dict.items():
                for det in dets:
                    det['track_id'] += track_id_offset_pred
        for frame_id, dets in gt_dict.items():
            merged_gt[frame_id+frame_id_offset].append(dets)
        for frame_id, dets in pred_dict.items():
            merged_pred[frame_id+frame_id_offset].append(dets)
        # Update offsets for next iteration
        frame_id_offset = max(max(merged_gt.keys(), default=0), max(merged_pred.keys(), default=0)) + 1
        track_id_offset_gt += max([det['track_id'] for dets in gt_dict.values() for det in dets], default=0) + 1
        track_id_offset_pred += max([det['track_id'] for dets in pred_dict.values() for det in dets], default=0) + 1
    # Save merged files
    merged_gt_path = os.path.join(output_folder, 'merged_gt')
    merged_pred_path = os.path.join(output_folder, 'merged_pred')
    save_mot_format(merged_gt, merged_gt_path)
    merged_pred_file = save_mot_format(merged_pred, merged_pred_path)
    return merged_gt_path, merged_pred_file

def merge_coco_formats(gt_files, detection_files, output_folder):
    '''
    Merge multiple COCO-format ground truth and detection files into a single COCO-format files to be able to evaluate
    a whole dataset at once. The merged files will be saved in the output_folder.
    Both are needed in order to be able to modify properly the image_id and annotation_id in the merged files.

    Args:
        gt_files: list of str, paths to the ground truth JSON files in COCO format
        detection_files: list of str, paths to the detection JSON files in COCO format
        output_folder: str, path to the folder where the merged files will be saved

    Returns:
        tuple of str, paths to the merged ground truth and detection JSON files in COCO format
    '''
    if len(gt_files) != len(detection_files):
        raise ValueError("The number of ground truth files and detection files must be the same.")
    merged_gt = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    merged_dt = []
    image_id_offset = 0
    annotation_id_offset = 0
    track_id_offset = 0
    for gt_file, dt_file in zip(gt_files, detection_files):
        gt_data = load_json_file(gt_file)
        dt_data = load_json_file(dt_file)
        # Update image_id and annotation_id in gt_data
        for img in gt_data['images']:
            img['id'] += image_id_offset
            merged_gt['images'].append(img)
        for ann in gt_data['annotations']:
            ann['id'] += annotation_id_offset
            ann['image_id'] += image_id_offset
            merged_gt['annotations'].append(ann)
        # Update image_id in dt_data
        for det in dt_data:
            det['image_id'] += image_id_offset
            if 'track_id' in det['attributes']:
                det['attributes']['track_id'] += track_id_offset
            merged_dt.append(det)
        # Update offsets for next iteration
        image_id_offset += len(gt_data['images'])
        annotation_id_offset += len(gt_data['annotations'])
        track_id_offset += max([det['attributes'].get('track_id', 0) for det in dt_data if 'track_id' in det['attributes']], default=0) + 1
    # Merge categories (assuming they are the same across all files)
    merged_gt['categories'] = gt_data['categories']
    # Save merged files
    os.makedirs(output_folder, exist_ok=True)
    merged_gt_file = os.path.join(output_folder, 'merged_gt.json')
    merged_dt_file = os.path.join(output_folder, 'merged_dt.json')
    save_json_file(merged_gt, merged_gt_file)
    save_json_file(merged_dt, merged_dt_file)
    return merged_gt_file, merged_dt_file

'''
Global Evaluation Functions
'''
def merge_eval_coco(video_outputs, eval_file, gt_file_name, preds_folder_name, output_merges, ignore_noid=False, cm=False, pr=False, visu=False, log=None):
    gt_file_per_video = {}
    pred_files_per_method_per_video = {}
    for video_output in video_outputs:
        gt_file = os.path.join(video_output, gt_file_name)
        if not os.path.exists(gt_file):
            print_and_log('No ground truth file found for video %s. Skipping evaluation for this video.' % (video_output), log=log)
            continue
        pred_folders = [
            os.path.join(video_output, preds_folder_name, f) for f in os.listdir(os.path.join(video_output, preds_folder_name)) \
                if os.path.isfile(os.path.join(video_output, preds_folder_name, f, 'detections.json'))
        ]
        gt_file_per_video[video_output] = gt_file
        for pred_folder in pred_folders:
            method_name = os.path.basename(pred_folder)
            if method_name not in pred_files_per_method_per_video:
                pred_files_per_method_per_video[method_name] = {}
            pred_files_per_method_per_video[method_name][video_output] = os.path.join(pred_folder, 'detections.json')
    for method_name in pred_files_per_method_per_video:
        # Skip methods that do not have predictions for all videos
        if len(pred_files_per_method_per_video[method_name]) != len(gt_file_per_video):
            print_and_log('Method %s does not have predictions for all videos. Skipping evaluation for this method.' % (method_name), log=log)
            continue
        gt_file, method_pred = merge_coco_formats(
            gt_file_per_video.values(),
            pred_files_per_method_per_video[method_name].values(),
            os.path.join(output_merges, '%s_merged.json' % (method_name)),
        )
        categories = {c['id']: c['name'] for c in load_json_file(gt_file)['categories']}
        if ignore_noid:
            # Get the key of the "NoID" class from categories
            ignore_classes = [k for k, v in categories.items() if v == 'NoID']
        else:
            ignore_classes = []
        eval_results = evaluate_detection(
            gt_file,
            method_pred,
            name='%s' % (method_name),
            save_path=eval_file,
            ignore_classes=ignore_classes,
            cm=cm,
            pr=os.path.join(os.path.dirname(eval_file), 'precision_recall', '%s.png' % (method_name)) if pr else '',
            visu=os.path.join(os.path.dirname(eval_file), 'visualizations', '%s' % (method_name)) if visu else '',
            log=log
        )
        if cm:
            cm_path = os.path.join(os.path.dirname(eval_file), 'confusion_matrices', '%s.png' % (method_name))
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            conf_matrix = eval_results['cm'][0]
            c_idxs = eval_results['cm'][1]
            plot_confusion_matrix(conf_matrix, [categories[i] if i in categories else i for i in c_idxs], cm_path)
        print_and_log('\tMethod %s: %s' % (method_name, str({k: v for k, v in eval_results.items() if k != 'cm'})), log=log)