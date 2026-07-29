helptext_input_video = '''Folder or file containing the video or images to process. 
The application supports folder and zip files containing '.jpeg','.jpg','.png','.bmp' and'.tif' files.
'''

helptext_output = '''Folder to save the outputs.
The detected Baboons will be saved in a .json file.
The images with the detected Baboons will be saved in a folder.
The video with the detected Baboons will be saved in a .mp4 file if ffmpeg is installed.
'''

helptext_video_demo = '''If this option is selected, the application will save the images and the video with the detected Baboons.
'''

helptext_max_res = '''Output max resolution in one dimension in pixels. Default: 720.
The resolution will be with respect to the largest dimension of the image to maintain the aspect ratio.
A higher resolution will increase the computation time.
'''

helptext_det_score_th = '''Detection threshold. Default: 0 to consider all detections. Max: 1.
The higher the value, the more likely the detections will be correct.
'''

helptext_del_imgs = '''Remove the images created for the video.
The images are saved in the output folder and are used to create the video.
This option is useful when the images are not needed and you want to save disk space.
'''

helptext_device = '''Device to use for the detection. Default: "cuda" if a GPU is available, otherwise "cpu".
Using "cuda" will significantly speed up the detection if a compatible GPU is available.
'''

helptext_tracking_size = '''Buffer size for tracking in number of frames. Default: 30.
The tracking buffer size determines how many frames the tracker will use to maintain the identity of detected Baboons across frames.
A larger buffer size can help maintain tracking accuracy in cases of occlusion or missed detections, but it may also lead to wrong associations.
Adjust this parameter based on the expected movement and density of Baboons in the video.
'''

helptext_gui = '''Use the GUI to select the parameters and visualize the results in real-time.
'''

helptext_tracker_type = '''Type of tracker to use. Default: None.
Options:
- "bytetrack": Use ByteTrack for tracking. This is a high-performance tracker that can handle occlusions and missed detections well.
- "deepsort": Use DeepSORT for tracking. This tracker uses a combination of motion and appearance features for tracking,
    which can be effective in crowded scenes.
- "botsort": Use BoTSORT for tracking. This tracker is designed for real-time applications and can handle occlusions and missed detections.
- "sam3": Use SAM 3 for tracking. This tracker uses a segmentation-based approach for tracking,
    which can be effective in cases of significant appearance changes or occlusions.
- None: Tracking based on IoU.
'''

helptext_eval_detection = '''Evaluate the detection performance using COCO metrics.
'''

helptext_eval_tracking = '''Evaluate the tracking performance using TrackEval metrics.
'''

helptext_eval_classification = '''Evaluate the classification performance using standard metrics.
'''

helptext_det_model = '''Model to use for detection. Default: "md_v5b.0.0.pt".
Options:
- "MDv5a": MegaDetector model trained on a dataset of animals in the wild.
- "MDv5b": Same but different training hyperparameters.
- path to a custom model: You can provide the path to a custom model trained on your own dataset following the same format as the MegaDetector models.
    The model should be a .pt file containing the weights of the model.
- "sam3": Use SAM 3 for detection and tracking.
- "sam3_det": Use SAM 3 for detection only.
'''

helptext_text_prompt = '''Text prompt to use for SAM 3 tracking. Default: "an animal".
This prompt will be used to guide the SAM 3 tracker in identifying and tracking the Baboons in the video.
You can experiment with different prompts to see how it affects the tracking performance.
'''

helptext_chunk_size = '''Chunk size for processing the video in segments with SAM3. Default: 200.
When using GPU, the entire chunk if loaded into memory.
An estimated chunk size can be calculated based on the available GPU memory and the average memory usage per frame.
Approximately: chunk_size ≈ (available_memory_GB − 5) × 100,000,000 / num_pixels
'''

helptext_overlap = '''Overlap size for processing the video in segments with SAM3. Default: 5.
The overlap size determines how many frames will be shared between consecutive chunks of the video.
This can help maintain tracking continuity across chunk boundaries, especially in cases where Baboons may move in and out of the frame.
A larger overlap may improve tracking accuracy but will increase processing time.
'''

helptext_class_det = '''Type of detector to use for classification. Default: None.
Options:
- "primateface": Use PrimateFace for detection. This detector is specifically designed for detecting primate faces and can be effective
    in identifying Baboons in the video.
- None: No detector will be used for classification. The classifier will operate on the bboxes provided by the detection step without any
    additional filtering or refinement.
'''

helptext_class_det_thr = '''Detection threshold for the classification detector. Default: 0.5.
This threshold determines the confidence level required for the classification detector to consider a detection valid.
A higher threshold will result in fewer detections being considered, which may reduce false positives but could also miss some true positives.
Adjust this parameter based on the expected quality of detections and the desired balance between precision and recall.
'''

helptext_class_nms_thr = '''Non-maximum suppression (NMS) threshold for the classification detector. Default: 0.4.
NMS is a technique used to eliminate redundant overlapping detections by keeping only the detection with the highest confidence score.
The NMS threshold determines how much overlap is allowed between detections before they are suppressed.
A lower threshold will result in more aggressive suppression, which can help reduce false positives but may also eliminate some true positives.
Adjust this parameter based on the expected density of Baboons in the video and the desired balance between precision and recall.
'''

helptext_avg_score = '''Use the average score for each class when classifying the detected Baboons. Default: False.
When enabled, the classifier will compute the average score of the best score retrieved for each sample in our datavase over the track.
'''

helptext_sim_th = '''Threshold for class assignment using similarity. Default: 0.5.
This threshold determines the minimum similarity score required for a detected Baboon to be assigned to a class.
A higher threshold will result in more conservative class assignments, potentially reducing false positives but increase the percentage of unclassified Baboons.
'''

helptext_feat_avg = '''Use the average features for each class when classifying the detected Baboons. Default: False.
When enabled, the classifier will compute the average feature vector for each class based on the training data and average the features of each
track. This can help improve classification accuracy, especially in cases where the individual feature vectors may be noisy or inconsistent.
'''

helptext_nca = '''Use Neighborhood Component Analysis (NCA) for classification. Default: False.
NCA is a dimensionality reduction technique that can be used to improve classification performance by learning a transformation of the feature space
that maximizes the separation between classes. When enabled, the classifier will apply NCA to the feature vectors computed from the datavase and learn
the projection that will be applied to the features of the detected Baboons before classification,
'''

helptext_epochs = '''Number of epochs for training the classifier. Default: 100.
'''

helptext_lr = '''Learning rate for training the classifier. Default: 1e-4.
'''

helptext_roi_factor = '''Region of interest (ROI) factor for increasing or decreasing the size of the bounding boxes used for classification. Default: 1.0.
This factor is used to scale the bounding boxes of the detected Baboons before they are passed to the classifier. A factor greater than 1.0 will increase
the size of the bounding boxes, potentially including more context around the Baboons (or its face), while a factor less than 1.0 will decrease the size of
the bounding boxes.
'''

helptext_roi_det = '''Region of interest (ROI) factor for increasing or decreasing the size of the bounding boxes used for detection. Default: 1.0.
'''

helptext_joint_factor = '''Factor for combining the scores from two classifiers. Default: 0
This factor determines how much weight is given to the scores from the second classifier when combining them with the scores from the first classifier.
'''

helptext_loop = '''Loop the processing of the video trying different parameters. This can be useful to find the best parameters for a video.
'''

helptext_class_database = '''Path to the classification dictionary. Default: "/shared/group_dict".
The classification dictionary is a folder containing subfolders for each class, and each subfolder contains images of that class.
The classification model will be trained on these images to classify the detected Baboons into the classes defined in the dictionary.
'''

helptext_save_mot = '''Save the results in MOT format. This can be useful for uploading the results to an annotation tool like CVAT or for further analysis.
'''

helptext_num_workers = '''Number of workers to use for parallel processing. Default: 0 (no parallel processing).
This parameter determines how many worker processes will be used to process the videos in parallel.
However, using multiple workers can lead to increased memory usage and potential issues with GPU memory allocation, especially when using CUDA.
'''