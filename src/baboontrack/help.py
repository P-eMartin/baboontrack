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

helptext_det_score = '''Baboon detection threshold. Default: 0 to consider all detected Baboons. Max: 1.
The higher the value, the more likely the detected Baboons will be correct.
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
- "deepsort": Use DeepSORT for tracking. This tracker uses a combination of motion and appearance features for tracking, which can be effective in crowded scenes.
- None: Tracking based on IoU.
'''