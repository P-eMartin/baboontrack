
import os
import cv2
import numpy as np
import zipfile
import natsort # type: ignore
from .io_utils import print_and_log
from .json_utils import load_json_file
from .ffmpeg_utils import get_ffmpeg_codec, create_video
import time
import subprocess
import re
import shutil

from .io_utils import progress_bar

'''
Video iterator
'''
class VideoFrameIterator:
    '''
    An iterator to iterate over the frames of a video file or image files.
    It supports both video files and image files such as .jpg, .png, .bmp, .tif, .zip files containing images.
    It can return a specific frame or iterate over all the frames.
    The iterator is reset automatically when the end of the video is reached.
    It contains information about the video such as fps, length, and whether it is a video or image files.
    It is advised to use the check_video method to check if the video is correctly loaded and to get the fps and length information.
    '''
    def __init__(self, path, video_extensions=('.mp4','.mkv','.avi','.mov', '.wmv'), img_extensions=('.jpeg','.jpg','.png','.bmp','.tif'), bgr=False, fps=None, log=None):
        '''
        Initialize the VideoFrameIterator.

        Args:
            path: str, the path to the video file or image file or list of files
            video_extensions: tuple, the video file extensions to consider (default: .mp4, .mkv, .avi, .mov, .wmv)
            img_extensions: tuple, the image file extensions to consider (default: .jpeg, .jpg, .png, .bmp, .tif)
            bgr: bool, whether to convert the frames to BGR format (default False, which converts to RGB)
            fps: float, the frames per second used when processing image files (default None)
            log: logging.Logger, the logger to log the information (default None)

        Returns:
            None
        '''
        self.path = path
        self.log = log
        self.bgr = bgr
        self.fps = fps
        # Check if the video path is a video file or directory/zip with image files
        if path.lower().endswith(video_extensions):
            self.cap = cv2.VideoCapture(path)
            # This may not work for all video formats
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.img = False
        else:
            # Create the list of image files to be processed
            if zipfile.is_zipfile(path):
                self.zip_file = zipfile.ZipFile(path, 'r')
                self.img_files = [f.filename for f in self.zip_file.infolist() if (f.filename.endswith(img_extensions)) and not os.path.basename(f.filename).split('/')[-1].startswith('.')]
            elif os.path.isdir(path):
                self.zip_file = None
                self.img_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(img_extensions) and not f.startswith('.')]
            else:
                raise ValueError("The path %s is not a valid video file, or zip file or directory containing image files." % path)
            # Sort the files using natsort
            self.img_files = natsort.natsorted(self.img_files)
            self.length = len(self.img_files)
            self.img = True
            if self.fps is None:
                print_and_log('Cannot get fps from %s.' % (path), log=log)
        self.idx = 0
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.img:
            if self.idx >= self.length:
                # raise StopIteration
                self.reset_and_stop()
            frame = cv2.imread(self.img_files[self.idx])
        else:
            if not self.cap.isOpened():
                # raise StopIteration
                self.reset_and_stop()

            ret, frame = self.cap.read()
            self.count += 1
            # Stop iteration when frame is not captured
            if not ret:
                if self.idx >= 0 or self.count >= self.length:
                    self.reset_and_stop()
                    # self.cap.release()
                    # raise StopIteration
                else:
                    # Call __next__ again to get the next frame
                    frame = self.__next__()
            # Convert the frame to RGB if needed (default)
            if not self.bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.idx += 1
        return frame

    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.img:
            if self.zip_file:
                self.zip_file.close()
        else:
            if self.cap.isOpened():
                self.cap.release()

    def check_video(self):
        '''
        Test the iterator by iterating over the video frames.
        This method is useful to check if the video is correctly loaded.
        '''
        start_time = time.time()
        for idx, frame in enumerate(self):
            if idx % 100 == 0:
                print_and_log("Frame %d/%d" % (idx, self.length), log=self.log)
        total_frames = idx + 1
        print_and_log("Video %s with fps %.2f iterated in %.2f seconds with a total frame of %d. FPS: %.2f" % (self.path, self.fps, time.time() - start_time, total_frames, total_frames / (time.time() - start_time)), log=self.log)
        if self.length != total_frames:
            print_and_log("Warning: The number of frames iterated (%d) is different from the video length (%d). Modifying video length information." % (total_frames, self.length), log=self.log)
            self.length = total_frames
            # check fps
            fps_ffmpeg = self.get_fps_ffmpeg()
            if fps_ffmpeg is not None and abs(fps_ffmpeg - self.fps) > 0.1:
                print_and_log("Warning: The fps of the video (%f) is different from the one retrieved by ffmpeg (%f). Modifying fps information." % (self.fps, fps_ffmpeg), log=self.log)
                self.fps = fps_ffmpeg
            elif fps_ffmpeg is None:
                print_and_log("Warning: Cannot retrieve fps from ffmpeg. Checking duration instead.", log=self.log)
                duration_ffmpeg = self.get_duration_ffmpeg()
                if duration_ffmpeg is not None:
                    calculated_fps = self.length / duration_ffmpeg
                    if abs(calculated_fps - self.fps) > 0.1:
                        print_and_log("Warning: The fps of the video (%f) is different from the one calculated from the duration (%f). Modifying fps information." % (self.fps, calculated_fps), log=self.log)
                        self.fps = calculated_fps
                else:
                    print_and_log("Warning: Cannot retrieve duration from ffmpeg. The fps information will not be modified.", log=self.log)
            print_and_log("After check, considering video %s with fps %.2f with a total frame of %d." % (self.path, self.fps, self.length), log=self.log)
    
    def reset_and_stop(self):
        print_and_log("Iterations: %d, Frames retrieved: %d Video length: %d" % (self.count, self.idx, self.length), log=self.log)
        self.reset_video()
        raise StopIteration
    
    def reset_video(self):
        '''
        Reset the video to the beginning.
        '''
        self.idx = 0
        self.count = 0
        if not self.img:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_duration_ffmpeg(self):
        # Run the ffmpeg command to get video stream information
        result = subprocess.run(
            ['ffmpeg', '-i', self.path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Decode the output and search for the duration information
        output = result.stderr.decode('utf-8')
        match = re.search(r'Duration: (\d{2}:\d{2}:\d{2}\.\d{2})', output)
        if match:
            duration = match.group(1)
            # Convert duration to seconds
            h, m, s = map(float, duration.split(':'))
            duration = h * 3600 + m * 60 + s
            return duration
        return None

    def get_fps_ffmpeg(self):
        # Run the ffmpeg command to get video stream information
        result = subprocess.run(
            ['ffmpeg', '-i', self.path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Decode the output and search for the FPS information
        output = result.stderr.decode('utf-8')
        for line in output.split('\n'):
            if 'fps' in line:
                # Extract the FPS value from the line
                parts = line.split(',')
                for part in parts:
                    if 'fps' in part:
                        fps = part.split('fps')[0].strip()
                        return float(fps)
        return None

    def get_fps(self):
        return self.fps
    
    def get_frame(self, frame_number):
        '''
        Get a frame from the video at the given frame number.
        This method is not robust to all video formats.
        
        Args:
            frame_number: int, the frame number to get
            
        Returns:
            numpy array: the frame at the given frame number
        '''
        if frame_number < 0 or frame_number >= self.length:
            print_and_log("Frame %d number out of range for video of length %d" % (frame_number, self.length), log=self.log)
            return None
        if self.img:
            frame = cv2.imread(self.img_files[frame_number])
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret:
                print_and_log("Error reading frame %d for video of length %d" % (frame_number, self.length), log=self.log)
                return None
            if not self.bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def plot_annotations(self, annotations, output_path, max_res=None, thickness=None, fontscale=None, display_fct=None, classification_classes=None, detection_classes=None, bbox_format='xywh', bbox_normalized=True, font=cv2.FONT_HERSHEY_SIMPLEX, del_imgs=False, log=None):
        '''
        Plot the annotations on the video frames and save the video.

        Args:
            annotations: list of dicts, the annotations to plot. Each dict should contain 'bbox', 'track_id', 'id', 'id_score',  'det' and 'det_score' keys.
            output_path: str, the path to save the annotated video
            max_res: int, the maximum resolution of the output video (default None)
            bbox_color_dict: dict, the color to use for plotting the annotations (default None)
            thickness: int, the thickness of the annotation lines (default 1)
            fontscale: float, the scale of the font (default None)
            display_fct: function, a function to call to display the frames (default None)
            classification_classes: list, the classification classes (default None)
            detection_classes: list, the detection classes (default None)
            bbox_format: str, the format of the bounding box coordinates (default 'xywh')
            bbox_normalized: bool, whether the bounding box coordinates are normalized (default True)
            font: int, the font to use for the text (default cv2.FONT_HERSHEY_SIMPLEX)
            del_imgs: bool, whether to delete the annotated images after creating the video (default False)
            log: logging.Logger, the logger to log the information (default None)

        Returns:
            int: 1 if the video was successfully saved, 0 otherwise
        '''
        # Check if output video exists
        if os.path.exists(output_path):
            print_and_log("Output video %s already exists. Skipping plotting." % (output_path), log=log)
            return 1
        
        # Create folder to save the annotated frames
        output_folder = os.path.splitext(output_path)[0]
        os.makedirs(output_folder, exist_ok=True)

        # Reset video
        self.reset_video()

        # Color dictionaries
        ## Detection classes based on viridis colormap
        if detection_classes is not None:
            detection_color_dict = get_colormap_dict(detection_classes, cv2.COLORMAP_VIRIDIS)
        else:
            detection_color_dict = None
        ## Classification classes based on turbo colormap
        if classification_classes is not None:
            classification_color_dict = get_colormap_dict(classification_classes, cv2.COLORMAP_TURBO)
        else:
            classification_color_dict = None

        # Check if the number of images in the folder is the same as the video length
        if len(os.listdir(output_folder)) != self.length:
            print_and_log("Plotting annotations on video %s..." % (self.path), log=log)
            start_time = time.time()
            # Loop over the video frames and plot the annotations
            for idx, frame in enumerate(self):
                elapsed_time = time.time() - start_time
                progress_bar(
                    idx,
                    self.length, 
                    title="Plotting annotations on video %s%s" % (
                        os.path.basename(self.path),
                        " (%ds left)" % (elapsed_time/idx*(self.length-idx-1)) if idx > 0 else ""
                    ), 
                    log=log
                )
                ## Resize wrt maximum resolution
                if max_res is not None:
                    h, w = frame.shape[:2]
                    if max(h, w) > max_res:
                        scale = max_res / max(h, w)
                        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                image_size = frame.shape[:2]
                ## Set thickness and fontscale based on the image size
                if idx == 0:
                    if thickness is None:
                        thickness = max(1, int(round(0.001 * (image_size[0] + image_size[1]) / 2)))
                    if fontscale is None:
                        fontscale = max(0.35, 0.0005 * (image_size[0] + image_size[1]) / 3)                
                ## Plot the annotations on the frame
                frame_annotations = annotations[idx] if idx < len(annotations) else []
                for ann in frame_annotations:
                    bbox = get_bbox(ann['bbox'], bbox_format=ann.get('bbox_format', 'xywh'), bbox_normalized=ann.get('bbox_normalized', True), image_size=image_size)
                    track_id = ann['track_id']
                    det = detection_classes[ann['det']] if detection_classes else ann['det']
                    det_score = ann['det_score']
                    class_id = classification_classes[ann['id']] if classification_classes else ann['id']
                    id_score = ann['id_score']
                    if detection_color_dict is not None:
                        color_det = detection_color_dict[det]
                    else:
                        color_det = (0, 255, 0) # default color is green
                    if classification_color_dict is not None:
                        color_id = classification_color_dict[class_id]
                    else:
                        color_id = (255, 0, 0) # default color is blue
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_det, thickness)
                    # Det score and Track ID - top of the box
                    text = ('%s (%.2f) Track %d' % (det, det_score, track_id)).replace('(0.', '(.').replace('(-0.', '(-.')
                    spacing = int(thickness*1.5)
                    txt_w, txt_h = write_text(frame, text, (bbox[0], bbox[1] - spacing), fontscale, color_bg=color_det, thickness=thickness, font=font)
                    # Class ID and Class ID score - above the det text
                    text = ('%s (%.2f)' % (class_id, id_score)).replace('(0.', '(.').replace('(-0.', '(-.')
                    write_text(frame, text, (bbox[0], bbox[1] - 4*spacing - txt_h), fontscale, color_bg=color_id, thickness=thickness, font=font)
                # Save the annotated frame
                cv2.imwrite(os.path.join(output_folder, '%d.png' % idx), frame)
                # Call the display function if provided
                if display_fct is not None:
                    display_fct(frame)
            progress_bar(self.length, self.length, title="Plotting annotations on video %s... Done in %.2f seconds." % (os.path.basename(self.path), time.time() - start_time), completed=True, log=log)
        else:
            print_and_log("Annotated frames already exist for video %s. Skipping plotting." % (self.path), log=log)

        create_video(
            os.path.join(output_folder),
            output_path,
            fps=self.fps,
            threads=0,
            codec=get_ffmpeg_codec(log=log),
            log=log,
        )
        if del_imgs:
            # Use shutil.rmtree to delete the folder checking it contains only expected images to avoid deleting important files by mistake
            if os.path.exists(output_folder) and os.path.isdir(output_folder):
                files = os.listdir(output_folder)
                if all(f.endswith('.png') for f in files):
                    shutil.rmtree(output_folder)
                else:
                    print_and_log("Warning: The folder %s contains files that are not .png images. The folder will not be deleted." % (output_folder), log=log)
            else:
                print_and_log("Warning: The folder %s does not exist or is not a directory. The folder will not be deleted." % (output_folder), log=log)

        # Reset video
        self.reset_video()
        return 1
    
def get_bbox(bbox, bbox_format='xywh', bbox_normalized=True, image_size=None):  
    '''
    Get the bounding box coordinates in xyxy format.

    Args:
        bbox: list, the bounding box coordinates
        bbox_format: str, the format of the bounding box coordinates (default 'xywh')
        bbox_normalized: bool, whether the bounding box coordinates are normalized (default True)
        image_size: tuple, the size of the image (height, width) (default None, required if bbox_normalized is True)

    Returns:
        list: the bounding box coordinates in xyxy format
    '''
    if bbox_format == 'xywh':
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
    elif bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError("Invalid bbox_format %s. Should be 'xywh' or 'xyxy'." % (bbox_format))
    if bbox_normalized:
        if image_size is None:
            raise ValueError("image_size should be provided when bbox_normalized is True.")
        x1 = int(x1 * image_size[1])
        y1 = int(y1 * image_size[0])
        x2 = int(x2 * image_size[1])
        y2 = int(y2 * image_size[0])
    return [x1, y1, x2, y2]

    
def write_text(frame, text, position, fontscale, color_text=None, color_bg=None, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    '''
    Write text on a frame with a background.

    Args:
        frame: numpy array, the frame to write on
        text: str, the text to write
        position: tuple, the position to write the text (x, y)
        fontscale: float, the scale of the font
        color_text: tuple, the color of the text (default None, which is white if color_bg is None or dark, and black if color_bg is light)
        color_bg: tuple, the color of the background (default None, which is transparent)
        thickness: int, the thickness of the text (default 1)
        font: int, the font to use (default cv2.FONT_HERSHEY_SIMPLEX)
    Returns:
        numpy array: the frame with the text written on it
    '''
    (text_width, text_height), baseline = cv2.getTextSize(text, font, fontscale, thickness)
    x, y = position
    
    if color_bg is not None:
        # Rectangle coordinates
        ## Add a bit of padding for visual centering and adjust y to better center the text vertically
        pad = int(0.1 * text_height)
        rect_x1 = x - pad
        rect_y1 = y - text_height - baseline - pad
        rect_x2 = x + text_width + pad
        rect_y2 = y + pad
        text_height+= 2*pad
        y-= int(baseline/2)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color_bg, -1)
    if color_text is None:
        if color_bg is not None:
            # If the background color is light, use black text, otherwise use white text
            if np.mean(color_bg) > 127:
                color_text = (0, 0, 0)
            else:
                color_text = (255, 255, 255)
        else:
            color_text = (255, 255, 255)
    cv2.putText(frame, text, (x, y), font, fontscale, color_text, thickness)
    return text_width, text_height

    
def get_colormap_dict(classes, colormap=cv2.COLORMAP_VIRIDIS):
    '''
    Get a color dictionary for the given classes based on the given colormap.
    Args:
        classes: list, the list of classes to get the color dictionary for
        colormap: int, the OpenCV colormap to use (default cv2.COLORMAP_VIRIDIS)

    Returns:
        dict: the color dictionary for the given classes
    '''
    return dict(zip(classes,map(lambda c: tuple(map(int, c)),cv2.applyColorMap(np.linspace(0, 255, len(classes), dtype=np.uint8)[:, None],colormap).reshape(-1, 3))))

    
def get_trailer(input_path, N=3):
    '''
    Extract N frames from the input_path video as np arrays. Supports RGB and image files.

    Args:
        input_path: str, the path to the video file or zip/directory containing image files
        N: int, the number of frames to extract (default 3)

    Returns:
        list: the list of frames as np arrays
    '''
    # Check if input_path is a VideoFrameIterator
    if isinstance(input_path, VideoFrameIterator):
        my_video = input_path
        del_video = False
    else:
        my_video = VideoFrameIterator(input_path)
        del_video = True
    
    # Get idxs to return
    if N == 1:
        idxs_to_return = [len(my_video)//2]
    else:
        idxs_to_return = np.linspace(0,len(my_video)-1,N,dtype=int)

    imgs_list = []

    for idx in idxs_to_return:
        imgs_list.append(my_video.get_frame(idx))
            
    # Close zip file
    if del_video:
        del my_video

    return imgs_list
    
'''
Segmentation functions
'''
def get_value_from_contours(img_ori, pointsLoc=[], mask=None, maxVals_factor=0, minVals_factor=0, replicate_study=False):
    '''
    Get the average, min, max values and min, max location of an image.
    Optionally it can also returns the average of percentage of max and min values.
    It supports mask and contours (filled polygons) to get the values from a specific region.

    Args:
        img_ori: numpy array, the image to process (can be a RGG or grayscale frame)
        pointsLoc: list, the list of points to consider (default [])
        mask: numpy array, the mask to apply (default None)
        maxVals_factor: float, the factor to get the average of the max values (default 0)
        minVals_factor: float, the factor to get the average of the min values (default 0)

    Returns:
        meanVal: float, the average value of the image
        minVal: float, the minimum value of the image
        maxVal: float, the maximum value of the image
        minLoc: tuple, the location of the minimum value
        maxLoc: tuple, the location of the maximum value
        minVals_avg: float, the average of the min values (if minVals_factor is set)
        maxVals_avg: float, the average of the max values (if maxVals_factor is set)
    '''
    # Convert to grayscale if needed
    if len(img_ori.shape)==3 and img_ori.shape[2]>1:
        if replicate_study:
            '''
            From "Estimating the cardiac signals of chimpanzees using a digital camera:
            validation and application of a novel non‑invasive method for primate research"
            by Danyi Wang et al.:
            
            In opencv it is:
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cr = (R - Y) * 0.713 + delta
            Cb = (B - Y) * 0.564 + delta
            
            But in their paper:
            Y = 65.841 * R + 128.553 * G + 24.966 * B + 16
            Cb = -39.797 * R - 74.203 * G + 112 * B + 128
            Cr = 112 * R - 93.786 * G - 18.2214 * B + 128
            '''
            # Convert the image to Y component
            img = 65.841 * img_ori[:,:,2] + 128.553 * img_ori[:,:,1] + 24.966 * img_ori[:,:,0] + 16
        else:
            # Convert to grayscale
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)        
    else:
        img = img_ori.copy()
    if len(pointsLoc)==0 or len(pointsLoc[0])==0:
        mask = np.ones(img.shape, dtype=np.uint8)
    elif isinstance(pointsLoc[0][0], np.ndarray) or isinstance(pointsLoc[0][0], list) or isinstance(pointsLoc[0][0], tuple):
        # Case when pointsLoc is a list of contours
        if isinstance(pointsLoc[0], np.ndarray):
            mask = cv2.fillPoly(np.zeros(img.shape, np.uint8), pointsLoc, 1)
        else:
            mask = cv2.fillPoly(np.zeros(img.shape, np.uint8), [np.array(contour) for contour in pointsLoc], 1)
    else:
        # Case when pointsLoc is a single contour
        if isinstance(pointsLoc, np.ndarray):
            mask = cv2.fillPoly(np.zeros(img.shape, np.uint8), [pointsLoc], 1)
        else:
            mask = cv2.fillPoly(np.zeros(img.shape, np.uint8), [np.array(pointsLoc)], 1)
    
    debug = False
    if debug:
        print('Debug mode for the mask')
        os.makedirs('mask_debug', exist_ok=True)
        mask_files = [f for f in os.listdir('mask_debug') if os.path.isfile(os.path.join('mask_debug', f)) and f.startswith('mask_')]
        cv2.imwrite('mask_debug/mask_%d.png' % len(mask_files), mask*255)
    
    # Get values
    meanVal = cv2.mean(img, mask=mask)[0]
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img, mask=mask)

    # # This alternative method is slower
    # start_time = time.time()
    # # Apply the mask to the image
    # masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # # Calculate mean, min, and max using masked image
    # meanVal2 = np.mean(masked_img[mask > 0])
    # minVal2 = np.min(masked_img[mask > 0])
    # maxVal2 = np.max(masked_img[mask > 0])
    # print('Time to get meanVal2, minVal2, maxVal2: %f' % (time.time()-start_time))

    # Save average of the maxVals_factor max values of the mask region
    if (0 < maxVals_factor < 1) or (0 < minVals_factor < 1):
        img_values = img[mask > 0].flatten()
        img_values.sort()
        n_values = len(img_values)

        # Deals with the case where the mask is empty
        maxVals_avg = img_values[-int(n_values*maxVals_factor+1):].mean() if n_values>0 else maxVal
        minVals_avg = img_values[:int(n_values*minVals_factor+1)].mean() if n_values>0 else minVal

        return meanVal, minVal, maxVal, minLoc, maxLoc, minVals_avg, maxVals_avg

    return meanVal, minVal, maxVal, minLoc, maxLoc
        
def get_contour_from_bbox(bbox, bbox_format='xywh'):
    '''
    Get the contour from a bbox. Can deal with xyxy or xywh methods.

    Args:
        bbox: list, the bounding box
        bbox_format: str, the format of the bounding box (default 'xywh')
    
    Returns:
        numpy array: the contour of the bounding box
    '''
    if bbox_format=='xywh':
        x1,y1,w,h = bbox
        x2 = x1 + w
        y2 = y1 + h
    elif bbox_format=='xyxy':
        x1,y1,x2,y2 = bbox
    return np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])

def enforce_min_bbox(bbox, frame_shape, min_size=50):
    """
    Ensure bbox is inside frame and has at least min_size.
    
    Args:
        bbox: [x1, y1, x2, y2]
        frame_shape: (H, W) or (H, W, C)
        min_size: minimum width/height
    
    Returns:
        [x1, y1, x2, y2] clipped and enlarged if needed
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    
    # 1) If bbox already valid, return as-is
    if x1 >= 0 and y1 >= 0 and x2 <= w-1 and y2 <= h-1 and (x2-x1) >= min_size and (y2-y1) >= min_size:
        return [x1, y1, x2, y2]
    
    # 2) Clamp bbox inside frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w-1, x2)
    y2 = min(h-1, y2)
    
    # 3) Compute width and height after clamping
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # 4) Enlarge if width or height < min_size
    if bbox_w < min_size or bbox_h < min_size:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_w = max(bbox_w, min_size) / 2
        half_h = max(bbox_h, min_size) / 2
        
        # Centered bbox
        x1_new = int(round(cx - half_w))
        x2_new = int(round(cx + half_w))
        y1_new = int(round(cy - half_h))
        y2_new = int(round(cy + half_h))
        
        # Shift bbox inside frame without unnecessary movement
        x1_new = max(0, x1_new)
        x2_new = min(w-1, x2_new)
        y1_new = max(0, y1_new)
        y2_new = min(h-1, y2_new)

        # If after clamping the size < min_size, expand on the side that allows it
        if (x2_new - x1_new) < min_size:
            # Expand left if possible
            x1_new = max(0, x2_new - min_size)
            # Expand right if possible
            x2_new = min(w-1, x1_new + min_size)

        if (y2_new - y1_new) < min_size:
            y1_new = max(0, y2_new - min_size)
            y2_new = min(h-1, y1_new + min_size)
        
        x1, x2, y1, y2 = x1_new, x2_new, y1_new, y2_new
    
    return [x1, y1, x2, y2]