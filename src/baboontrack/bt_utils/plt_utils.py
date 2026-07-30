import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import cv2
from .io_utils import progress_bar
from .ffmpeg_utils import create_video, get_ffmpeg_codec

def save_img_with_bbox(img, bbox, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x, y, w, h = bbox
    cv2.rectangle(img, (max(0, int(x)), max(0, int(y))), (min(int(x + w), img.shape[1]), min(int(y + h), img.shape[0])), (0, 255, 0), 2)
    cv2.imwrite(save_path, save_path)
    return 1


def draw_annotation(ax, annotation, annotation_key, color, fontsize=12, linewidth=1, fy=1):
    '''
    Draw annotation on a Matplotlib Axes. It can be a point, a contour, a box or a bbox.

    Args:
        ax: matplotlib.axes.Axes, the axes to draw the annotation on
        annotation: dict or list, annotation to draw
        annotation_key: str, key of the annotation
        color: color for the annotations
        fontsize: int, fontsize for the text
        linewidth: int, linewidth for the annotations
        fy: float, scaling factor for the annotation

    Returns:
        elements: list, a list of Matplotlib elements added to the axes
    '''
    elements = []  # List to store the elements plotted

    # Rescale annotation if needed
    if isinstance(annotation, dict):
        annotation = fy * np.array(list(annotation.values()))
    else:
        annotation = fy * np.array(annotation)

    # Check the type of annotation and draw it
    if annotation_key == 'bbox':
        rect = plt.Rectangle(tuple(annotation[:2]), annotation[2] - annotation[0], annotation[3] - annotation[1],
                             edgecolor='green', facecolor='none', linewidth=linewidth)
        ax.add_patch(rect)
        elements.append(rect)
    elif annotation_key == 'box':
        line, = ax.plot(annotation[:, 0], annotation[:, 1], color='blue', linewidth=linewidth)
        elements.append(line)
    else:
        # Location
        text = annotation_key.replace('_', ' ')
        if len(annotation) > 0:  # If annotation is not empty
            if isinstance(annotation[0], np.ndarray):  # Region
                loc_mean = annotation.mean(axis=0, dtype=int)
                # Drawing contours
                if isinstance(annotation[0][0], np.ndarray):  # List of contours
                    for contour in annotation:
                        loc_mean = contour.mean(axis=0, dtype=int)
                        polygon = plt.Polygon(contour, closed=True, edgecolor=color, facecolor='none', linewidth=linewidth)
                        polygon_patch = ax.add_patch(polygon)
                        elements.append(polygon_patch)
                        # line, = ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=linewidth)
                        # elements.append(line)
                        txt = ax.text(loc_mean[0], loc_mean[1], text, fontsize=fontsize, color=color, ha='center')
                        elements.append(txt)
                else:  # Single contour
                    loc_mean = annotation.mean(axis=0, dtype=int)
                    polygon = plt.Polygon(annotation, closed=True, edgecolor=color, facecolor='none', linewidth=linewidth)
                    polygon_patch = ax.add_patch(polygon)
                    elements.append(polygon_patch)
                    # line, = ax.plot(annotation[:, 0], annotation[:, 1], color=color, linewidth=linewidth)
                    # elements.append(line)
                    txt = ax.text(loc_mean[0], loc_mean[1], text, fontsize=fontsize, color=color, ha='center')
                    elements.append(txt)
            else:  # Point
                point, = ax.plot(annotation[0], annotation[1], 'o', color=color, markersize=4)
                elements.append(point)
                txt = ax.text(annotation[0], annotation[1], text, fontsize=fontsize, color=color, ha='center')
                elements.append(txt)

    return elements

def draw_annotation_on_img(img, annotation, annotation_key, colormap_annotation, fontface=cv2.FONT_HERSHEY_SIMPLEX, fontscale=.3, thickness=1, fy=1):
    '''
    Draw annotation on an image. It can be a point, a contour, a box or a bbox.

    Args:
        img: np.array, image to draw the annotation on
        annotation: dict or list, annotation to draw
        annotation_key: str, key of the annotation
        colormap_annotation: dict, colormap for the annotations
    '''
    # Rescale annotation
    if isinstance(annotation, dict):
        annotation = (fy*np.array(list(annotation.values()))).astype(int)
    else:
        annotation = (fy*np.array(annotation)).astype(int)
    # Check the type of annotation and draw it
    if annotation_key == 'bbox':
        cv2.rectangle(img, tuple(annotation[:2]), tuple(annotation[2:]), (0, 255, 0), 2) #drawing bbox in Green
    elif annotation_key == 'box':
        cv2.drawContours(img,[annotation],0,(255,0,0), 2) #drawing bbox in Blue
    else:
        # Location
        color = colormap_annotation[annotation_key]
        text = annotation_key.replace('_',' ')
        textsize = np.array(cv2.getTextSize(text, fontface, fontscale, thickness)[0])
        if len(annotation)>0: # If annotation is not empty (can be empty for various reasons)
            if isinstance(annotation[0],np.ndarray): # Region
                loc_mean = annotation.mean(axis=0, dtype=int)
                # Drawing contours
                if isinstance(annotation[0][0],np.ndarray): # List of contours
                    for contour in annotation:
                        loc_mean = contour.mean(axis=0, dtype=int)
                        cv2.drawContours(img, [contour], 0, color, thickness=2*thickness)
                        cv2.putText(img, text=text, org=loc_mean-textsize//2, fontFace=fontface, fontScale=fontscale, color=color, thickness=thickness)
                else: # Single contour
                    loc_mean = annotation.mean(axis=0, dtype=int)
                    cv2.drawContours(img, [annotation], 0, color, thickness=2*thickness) #drawing contours
                    cv2.putText(img, text=text, org=loc_mean-textsize//2, fontFace=fontface, fontScale=fontscale, color=color, thickness=thickness)
            else: # Point
                cv2.circle(img, annotation, radius=4, color=color, thickness=thickness)
                cv2.putText(img, text=text, org=annotation-textsize//2, fontFace=fontface, fontScale=fontscale, color=color, thickness=thickness)

def plot_axes_in_new_fig(axes):
    # Create a new figure and axis
    fig, ax_new = plt.subplots()

    # Extract images from the original axis
    images = axes.get_images()
    for image in images:
        img_data = image.get_array()
        ax_new.imshow(img_data, aspect=image.get_aspect(), extent=image.get_extent(), origin=image.get_origin())
        ax_new.set_xlim(image.get_extent()[0], image.get_extent()[1])
        ax_new.set_ylim(image.get_extent()[2], image.get_extent()[3])
        ax_new.set_aspect(image.get_aspect())
        ax_new.set_title(image.get_title())
        ax_new.set_xlabel(image.get_xlabel())
        ax_new.set_ylabel(image.get_ylabel())
    # Copy the axis limits
    ax_new.set_xlim(axes.get_xlim())
    ax_new.set_ylim(axes.get_ylim())
    # Copy the axis ticks
    ax_new.set_xticks(axes.get_xticks())
    ax_new.set_yticks(axes.get_yticks())
    ax_new.set_xticklabels(axes.get_xticklabels())
    ax_new.set_yticklabels(axes.get_yticklabels())

    # Extract data from the original axis
    lines = axes.get_lines()
    for line in lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        ax_new.plot(x_data, y_data, label=line.get_label())

    # Copy labels and title
    ax_new.set_xlabel(axes.get_xlabel())
    ax_new.set_ylabel(axes.get_ylabel())
    ax_new.set_title(axes.get_title())

    # Show legend if it exists
    if axes.get_legend():
        ax_new.legend()

    return fig

def plot_signals_video_different_fps(
        length_in_second,
        signals={},
        signals_title={},
        keys_tmp=[],
        keys_value=[],
        signals_peaks={},
        signals_mask={},
        skip_mask_keys=['score'],
        my_video=None,
        frame_annotations={},
        cmap_annotation= 'rainbow',
        cmap_value='brg',
        frame_normalization={'min':[0],'max':[255]},
        fontsize=12,
        fps_video=30,
        linewidth=1,
        im_w = 3,
        im_h = 3,
        col_sup=1,
        alpha_plot=.8,
        alpha_text=.6,
        save_path='plot_signals_video',
        secs_to_plot=6,
        suptitle='',
        height_video=1080,
        display_fct=None,
        check_fct=None,
        log=None,
    ):
    '''
    Plot signals and create video of animated signals.
    Can be coupled with frames and keypoints, contours or bbox.
    Signals should be provided as a dictionary with the key being the signal name and the value being the signal values.
    Supports different fps for the signals and the video.
    Peak and mask should be provided as a dictionary with similar keys as the signals.

    Args:
        length_in_second: int, length of the video in seconds
        signals: dict, signals that should be plot entirely - first key will be used as reference for the time
        keys_tmp: dict, signal keys of the signal dict that should be plot for a certain time
        keys_value: dict, signal keys of the signal dict that should be plot with a text
        signals_peaks: dict, peaks to plot
        signals_mask: dict, mask to plot
        my_video: video object, video object to use (default None)
        frame_annotations: dict, annotations to plot
        cmap_annotation: str, colormap for the annotations
        cmap_value: str, colormap for the values
        frame_normalization: dict, normalization for the frames
        fontsize: int, fontsize for the text
        fps_video: int, fps for the video
        fontface: int, fontface for the text
        fontscale: float, fontscale for the text
        thickness: int, thickness for the text
        ncols: int, number of columns for the subplots
        alpha_plot: float, alpha for the plot
        alpha_text: float, alpha for the text
        save_path: str, path to save the video
        secs_to_plot: int, seconds to plot
        suptitle: str, suptitle for the video
        height_video: int, height of the video
    '''
    # Initialization
    os.makedirs(os.path.join(save_path), exist_ok=True)
    n_annotations = len(list(frame_annotations.keys()))
    colormap_annotation = {annotation_key:matplotlib.cm.get_cmap(cmap_annotation, n_annotations)(idx) for idx, annotation_key in enumerate(frame_annotations.keys())}
    colormap_text = plt.get_cmap(cmap_value)
    if len(signals.keys()) > 0:
        # Convert to np array if needed
        for key in signals.keys():
            if not isinstance(signals[key], np.ndarray):
                signals[key]=np.array(signals[key])
    else:
        print('Nothing to plot')
        return 1
    ncols=im_w+col_sup
    nlines = int(np.ceil((len(signals.keys())+im_h*im_w)/ncols))
    fps_signals = {key:len(signals[key])/length_in_second for key in signals.keys()}
    # Define the ticks per fps
    x_ticks = {fps:np.arange(length_in_second*fps)/fps for fps in set(fps_signals.values())}

    # Set figure with 16:9 ration
    fig = plt.figure(figsize=[height_video*16/9/100, height_video/100])

    # Padding around the big plot
    fig.suptitle(suptitle, fontsize=fontsize)
    # Create a grid with 2 columns: first for the thermal img, second for the plots
    gs = matplotlib.gridspec.GridSpec(nlines, ncols, figure=fig)
    ## img in top left corner
    ax_img = fig.add_subplot(gs[:im_h, :im_w])
    ax_img.axis('off')
    ax_img.set_title('Thermal image in °C', fontsize=fontsize)

    # SVG or PNG
    img_extension = 'png'

    # Initialize some variables
    im_elements = []
    im = None
    scatter = {}
    ax = {}
    text_plot = {}
    padxy = {}
    txt_x = {}
    txt_y = {}
    means = {}
    stds = {}

    # Reset video
    if my_video is not None:
        my_video.reset_video()

    for plt_key in keys_value:
        txt_x[plt_key] = x_ticks[fps_signals[plt_key]][len(x_ticks[fps_signals[plt_key]])//2]
        txt_y[plt_key] = np.min(signals[plt_key])+(np.max(signals[plt_key])-np.min(signals[plt_key]))/2
        means[plt_key] = np.mean(signals[plt_key])
        stds[plt_key] = np.std(signals[plt_key])

    # Subplots
    for idx_key, plt_key in enumerate(signals.keys()):
        # Compute idx_subplot taking n_subplot_img into account
        idx_subplot = idx_key+im_w*(min(1+idx_key//(ncols-im_w),im_h))+1
        ax[plt_key] = plt.subplot(nlines, ncols, idx_subplot)
        if plt_key in signals_title:
            ax[plt_key].set_title(signals_title[plt_key], fontsize=fontsize)
        else:
            ax[plt_key].set_title(plt_key.replace('_',' '), fontsize=fontsize)
        ax[plt_key].plot(x_ticks[fps_signals[plt_key]][:len(signals[plt_key])], signals[plt_key], color='b', alpha=alpha_plot)
        ax[plt_key].set_xlim([x_ticks[fps_signals[plt_key]][0], x_ticks[fps_signals[plt_key]][-1]])
        if plt_key in signals_peaks.keys():
            # Plot peaks filtering out peaks that are out of the signal range
            peaks = [peak for peak in signals_peaks[plt_key] if peak<len(signals[plt_key])]
            ax[plt_key].scatter(x_ticks[fps_signals[plt_key]][peaks], signals[plt_key][peaks], color='red', marker='+', alpha=alpha_plot)
        
        if plt_key in keys_tmp:
            mask = None
        elif isinstance(signals_mask, dict):
            mask = np.array(signals_mask[plt_key]) if plt_key in signals_mask.keys() else None
        else:
            mask = np.array(signals_mask) if signals_mask is not None else None

        if mask is not None:
            ymin = np.min(signals[plt_key])
            ymax = np.max(signals[plt_key])
            yrange = ymax-ymin
            ax[plt_key].fill_between(x_ticks[fps_signals[plt_key]][:len(mask)], ymin-0.1*yrange, ymax+0.1*yrange, where=mask==0, facecolor='orange', alpha=0.3)
            if plt_key not in skip_mask_keys:
                # Max and min should be change according to when the mask is equal to one
                min_mask = np.min(signals[plt_key][mask==1])
                max_mask = np.max(signals[plt_key][mask==1])
                range_mask = max_mask-min_mask
                ax[plt_key].set_ylim([min_mask-0.1*range_mask, max_mask+0.1*range_mask])
                # Update position for text in the tmp plots
                if plt_key in keys_value:
                    txt_y[plt_key] = np.min(signals[plt_key][mask==1])+(np.max(signals[plt_key][mask==1])-np.min(signals[plt_key][mask==1]))/2
            
    # Create a video at fps_video regardless of the incoming fps
    frame = None
    start_time = time.time()
    for idx in range(int(length_in_second*fps_video)):
        # Early exit according to check function
        if check_fct is not None and check_fct(): return 0
        # Print progress but log only every 50 frames
        progress_bar(idx, int(length_in_second*fps_video), 'Plotting signals and creating video...', completed=0, log=log if idx % 50 == 0 else None)
        # Get index on a window time according to the fps
        idx_signal = {fps: min(int(idx/fps_video*fps), len(signals[plt_key])-1) for plt_key, fps in fps_signals.items()}
        start_plot = {fps: int(min(max(0, idx_signal[fps] - secs_to_plot*fps/2), len(signals[plt_key])-1-secs_to_plot*fps)) for plt_key, fps in fps_signals.items()}
        end_plot = {fps: int(start_plot[fps]+secs_to_plot*fps) for plt_key, fps in fps_signals.items()}
        
        # Subplots
        for plt_key in signals.keys():
            fps_signal = fps_signals[plt_key]
            # Update current position
            if idx == 0:
                scatter[plt_key] = ax[plt_key].scatter(x_ticks[fps_signal][idx_signal[fps_signal]], signals[plt_key][idx_signal[fps_signal]], color='orange', marker='o', alpha=alpha_plot)
            else:
                scatter[plt_key].set_offsets([x_ticks[fps_signal][idx_signal[fps_signal]], signals[plt_key][idx_signal[fps_signal]]])

            # Update limits
            if plt_key in keys_tmp:
                tmp_plot = signals[plt_key][start_plot[fps_signal]:end_plot[fps_signal]]
                tmp_min = np.min(tmp_plot)
                tmp_max = np.max(tmp_plot)
                val_range = tmp_max-tmp_min
                ax[plt_key].set_ylim([tmp_min-0.1*val_range, tmp_max+0.1*val_range])
                ax[plt_key].set_xlim([x_ticks[fps_signal][start_plot[fps_signal]], x_ticks[fps_signal][end_plot[fps_signal]]])
                # Update position for text in the tmp plots
                if plt_key in keys_value:
                    txt_x[plt_key] = x_ticks[fps_signal][int(max(secs_to_plot*fps_signal/2, min(len(signals[plt_key])-1-secs_to_plot*fps_signal/2, idx_signal[fps_signal])))]
                    txt_y[plt_key] = tmp_min+val_range/2
                # Update pads to avoid jittering when x or y ticks are updated. Still not perfect when ticks resolution changes on the y axis
                if idx == 0:
                    # Transparent text
                    padxy[plt_key] = ax[plt_key].text(x_ticks[fps_signal][end_plot[fps_signal]], tmp_max+0.2*val_range, '   ', fontsize=fontsize, color='white', alpha=0)
                else:
                    padxy[plt_key]._x = x_ticks[fps_signal][end_plot[fps_signal]]
                    padxy[plt_key]._y = tmp_max+0.2*val_range

            # Update text
            if plt_key in keys_value:
                color = colormap_text(max(0,min(1,(signals[plt_key][idx_signal[fps_signal]]-means[plt_key])/stds[plt_key])))
                txt_val = '%.3g' % (signals[plt_key][idx_signal[fps_signal]])
                if idx == 0:
                    text_plot[plt_key] = ax[plt_key].text(txt_x[plt_key], txt_y[plt_key], txt_val, ha='center', va='center', fontsize=3*fontsize, color=color, alpha=alpha_text)
                else:
                    text_plot[plt_key]._color = color
                    text_plot[plt_key]._text = txt_val
                    text_plot[plt_key]._x = txt_x[plt_key]
                    text_plot[plt_key]._y = txt_y[plt_key]            

        # Get new image otherwise use last one
        if my_video is not None and idx < len(my_video):
            # Could be modified so that the fps from the frames is used (here it assumes that the fps is the same as the video)
            # Get the frame from the iterator
            frame = my_video.__next__()

            ### Remove the previous image and its associated elements
            for el in im_elements:
                el.remove()
            im_elements = []
            if im is not None:
                im.remove()
            
            if my_video.ti:
                im = ax_img.imshow(frame, cmap='jet', vmin=frame_normalization['min'][idx] if idx<len(frame_normalization['min']) else frame_normalization['min'][-1], vmax=frame_normalization['max'][idx] if idx<len(frame_normalization['max']) else frame_normalization['max'][-1])
                ### Add colorbar to the image
                im_elements.append(plt.colorbar(im, ax=ax_img))
            else:
                im = ax_img.imshow(frame)
            ### Add the annotations
            for annotation_key in frame_annotations.keys():
                im_elements.extend(
                    draw_annotation(
                        ax_img,
                        frame_annotations[annotation_key][idx],
                        annotation_key,
                        colormap_annotation[annotation_key],
                        fontsize=0.8*fontsize,
                        linewidth=linewidth
                    )
                )

            ## Only on the first iteration
            if idx == 0:
                plt.tight_layout()                
                ## Set the limits of the image
                ax_img.set_xlim([0, frame.shape[1]])
                ax_img.set_ylim([frame.shape[0], 0])

        # Save figure
        save_path_img = os.path.join(save_path, '%d.%s' % (idx, img_extension))
        plt.savefig(save_path_img, bbox_inches='tight', transparent=True)

        if display_fct:
            # Faster plotting with saved image than with buffer image
            display_fct(save_path_img, text='%d/%d' % (idx, int(length_in_second*fps_video)), update_only=False if idx==0 else True, async_=True)

            # Transform figure to image
            # canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
            # canvas.draw()       # draw the canvas, cache the renderer
            # if Version(matplotlib.__version__) >= Version('3.4.0'):
            #     image_fig = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            #     # Do not forget alpha channel
            #     image_fig = image_fig.reshape(canvas.get_width_height()[::-1] + (4,))
            # else:
            #     image_fig = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            #     image_fig = image_fig.reshape(canvas.get_width_height()[::-1] + (3,))
            # display_fct(image_fig, text='%d/%d' % (idx, int(length_in_second*fps_video)), update_only=False if idx==0 else True)

    plt.close('all')
    # Progress bar
    progress_bar(idx+1, int(length_in_second*fps_video), 'Plotting signals and creating video completed in %.2f seconds' % (time.time()-start_time), completed=1, log=log)
    # Early exit according to check function
    if check_fct is not None and check_fct(): return 0
    create_video(
        os.path.join(save_path),
        save_path+'.mp4',
        fps=fps_video,
        sequence='%%d.%s' % (img_extension),
        threads=0,
        codec=get_ffmpeg_codec(log=log),
        log=log,
        # extra_args=['-vf', '"format=rgba, colorchannelmixer=aa=1.0, pad=iw:ih:color=white"'] if img_extension == 'svg' else []
    )

    # Reset video
    if my_video is not None:
        my_video.reset_video()

    return 1

def get_color_map(categories):
    '''
    Get a color map from a list of categories using the rainbow colormap.

    Args:
        categories: list, list of categories

    Returns:
        dict: dictionary of colors for each category
    '''
    # Dict of colors
    dict_colors = {cat:color for cat, color in zip(categories, matplotlib.cm.rainbow(np.linspace(0, 1, len(categories))))}
    return dict_colors

def generate_colormap_with_legend(min_val=0, max_val=1, width=50, height=720, fontsize=12):
    '''
    Generate a JET colormap with a legend for the values.
    
    Args:
        min_val: float, minimum value of the colormap
        max_val: float, maximum value of the colormap
        width: int, width of the colormap
        height: int, height of the colormap
        fontsize: int, fontsize of the legend
    
    Returns:
        np.array: the colormap with the legend
    '''
    # Create an image with the jet colormap
    colormap = np.linspace(1, 0, height).astype(np.float32)
    colormap = np.repeat(colormap[:, np.newaxis], width, axis=1)
    colormap_jet = cv2.applyColorMap((colormap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Calculate the interval for the legend based on height and fontsize
    num_steps = height // (fontsize * 3)  # Roughly 3 times the fontsize for each label
    step_value = (max_val - min_val) / num_steps
    # Find a step value with fewer digits
    rounded_step_value = round(step_value, -int(np.floor(np.log10(step_value))))
    
    # Draw the legend on the colormap
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(num_steps + 1):
        value = min_val + i * rounded_step_value
        # Ensure the value does not exceed max_val
        value = min(value, max_val)
        text = f"{value:.2f}"
        position = (5, int(height - (i * (height / num_steps))))
        cv2.putText(colormap_jet, text, position, font, fontsize / 30, (255, 255, 255), 1, cv2.LINE_AA)
    return colormap_jet


def plot_confusion_matrix(cm, classes, save_path, cmap=plt.cm.Blues, noid_name="NoID"):
    cm = np.asarray(cm)

    # Move NoID to the end
    if noid_name in classes:
        # Sorted order of classes with NoID at the end
        perm = [i for i, c in enumerate(classes) if c != noid_name]
        perm.append(classes.index(noid_name))
        cm = cm[np.ix_(perm, perm)]
        classes = [classes[i] for i in perm]

     # Normalization (row_wise)
    cm_normalized = (cm.T / np.maximum(cm.sum(axis=1), 1)).T

    # Accuracy
    if noid_name in classes:
        acc = np.trace(cm[:-1, :-1]) / np.sum(cm[:-1, :-1]) * 100 if np.sum(cm[:-1, :-1]) != 0 else 0
        acc_normalized = np.diag(cm_normalized[:-1, :-1])
    else:
        acc = np.trace(cm) / np.sum(cm) * 100 if np.sum(cm) != 0 else 0
        acc_normalized = np.diag(cm_normalized)

    mean_acc = np.mean(acc_normalized) * 100
    std_acc = np.std(acc_normalized) * 100

    title = 'Accuracy of %.3g%%%s\n$\\mu$ = %.3g with $\\sigma$ = %.3g' % (acc, ' (without noID)' if noid_name in classes else '', mean_acc, std_acc)

    n = len(classes)

    # TOTALS
    row_sum = cm.sum(axis=1)  # GT distribution
    col_sum = cm.sum(axis=0)  # predicted distribution

    total = np.sum(cm)

    # normalize totals for heatmap consistency
    row_sum_norm = row_sum / max(total, 1)
    col_sum_norm = col_sum / max(total, 1)

    # EXTENDED MATRIX (N+1)
    cm_ext = np.zeros((n + 1, n + 1), dtype=float)
    cm_ext[:n, :n] = cm_normalized

    # last column = GT totals
    cm_ext[:n, n] = row_sum_norm

    # last row = Pred totals
    cm_ext[n, :n] = col_sum_norm

    # bottom-right corner (global total)
    cm_ext[n, n] = 1.0  # represents 100% of dataset mass

    # FIGURE SIZE
    if n >= 12:
        plt.figure(figsize=(12, 12))
    elif n >= 6:
        plt.figure(figsize=(8, 8))
    else:
        plt.figure(figsize=(5, 5))

    ax = plt.gca()

    im = ax.imshow(cm_ext, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # LABELS
    ax.set_xticks(np.arange(n + 1))
    ax.set_yticks(np.arange(n + 1))

    ax.set_xticklabels(classes + ["TOTAL"], rotation=90, fontsize=14)
    ax.set_yticklabels(classes + ["TOTAL"], fontsize=14)

    ax.invert_yaxis()

    # TEXT ANNOTATION
    thresh = cm_ext.max() / 2.

    for i in range(n + 1):
        for j in range(n + 1):

            val = cm_ext[i, j]

            if i < n and j < n:
                text = f'{cm[i, j]}\n{val*100:.0f}%'

            elif i == n and j < n:
                text = f'{col_sum[j]}\n{val*100:.0f}%'

            elif j == n and i < n:
                text = f'{row_sum[i]}\n{val*100:.0f}%'

            else:
                text = f'{total}'

            ax.text(
                j, i, text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val > thresh else "black"
            )

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def plot_loss(losses, save_path, title='Loss', xlabel='Epochs', ylabel='Loss', fontsize=12):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Loss', color='blue')
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')