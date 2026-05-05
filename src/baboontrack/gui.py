import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
SYNC_WAIT_TIMEOUT = 3 # seconds
# One worker for display (module-level)
_display_executor = ThreadPoolExecutor(max_workers=1)
_display_future = None
GUI_PROCESSING = False
GUI_STOP_REQUESTED = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
import numpy as np
import copy
import os
import math
import random
import cv2
import traceback
import io
import datetime

# Utility functions
from .bt_utils.io_utils import print_and_log, setup_logger, close_log
from .bt_utils.img_utils import get_trailer
from .bt_utils.plt_utils import draw_annotation

# Help variables
from .help import *

def run_with_gui(args, main_function, check_args_fct=None):
    '''
    Run the application with a GUI.
    The GUI is used to set the parameters of the application and to display the results.
    
    Args:
        args: argparse.Namespace, the arguments
        main_function: function, the main function to run
        check_args_fct: function, the function to check the arguments (default None)
    '''

    # Some function variables
    ## GUI with height maximum and width with ratio 16:9
    max_height_gui = 600
    max_width_gui = max_height_gui*16/9
    max_height_input = 200
    max_width_input = max_width_gui
    max_height_logo = 100
    max_width_logo = 250
    
    class AppLog:
        '''
        Class to log within the GUI, the terminal and the log file.
        '''
        def __init__(self, terminal, terminal_size=(80,12), log_file=None):
            '''
            Initialize the AppLog class.
            
            Args:
                terminal: tk.Text, the terminal to log in
                log_file: str, the path to the log file
            '''
            self.terminal = terminal
            self.log_file = log_file
            self.terminal_size = terminal_size
            if log_file is not None:
                self.log = setup_logger(log_file=log_file)
                self.handlers = self.get_handlers()
            else:
                self.log = None

        def info(self, message):
            '''
            Log an info message.
            
            Args:
                message: str, the message to log
            '''
            ttk_terminal.config(state=tk.NORMAL)
            self.terminal.insert(tk.END, message + '\n')
            self.terminal.see(tk.END)
            ttk_terminal.config(state=tk.DISABLED)
            if self.log is not None:
                self.log.info(message)
            # Update the GUI
            self.terminal.update_idletasks()

        def get_handlers(self):
            '''
            Get the handlers of the logger.

            Returns:
                list, the handlers of the logger
            '''
            return self.log.handlers
        
        def removeHandler(self, handler):
            '''
            Remove a handler from the logger.

            Args:
                handler: logging.Handler, the handler to remove
            '''
            self.log.removeHandler(handler)

        def close(self):
            '''
            Close the log file.
            '''
            close_log(self.log)
    
    class CreateToolTip(object):
        '''
        Create a tooltip for a given widget.
        '''
        def __init__(self, widget, text='widget info', waittime=300, wraplength=400):
            '''
            Initialize the CreateToolTip class.

            Args:
                widget: tk.Widget, the widget to create a tooltip for
                text: str, the text of the tooltip
                waittime: int, waiting time before showing the tooltip in milliseconds (default: 300)
                wraplength: int, the width of the tipbox (default: 400)
            '''
            self.waittime = waittime
            self.wraplength = wraplength
            self.widget = widget
            self.text = text
            self.widget.bind("<Enter>", self.enter)
            self.widget.bind("<Leave>", self.leave)
            self.widget.bind("<ButtonPress>", self.leave)
            self.id = None
            self.tw = None

        def enter(self, event=None):
            '''
            Show the tooltip.

            Args:
                event: tk.Event, the event
            '''
            self.schedule()

        def leave(self, event=None):
            '''
            Hide the tooltip.

            Args:
                event: tk.Event, the event
            '''
            self.unschedule()
            self.hidetip()

        def schedule(self):
            '''
            Schedule the tooltip.
            '''
            self.unschedule()
            self.id = self.widget.after(self.waittime, self.showtip)

        def unschedule(self):
            '''
            Unschedule the tooltip.
            '''
            id = self.id
            self.id = None
            if id:
                self.widget.after_cancel(id)

        def showtip(self, event=None):
            '''
            Show the tooltip.

            Args:
                event: tk.Event, the event
            '''
            x = y = 0
            x, y, cx, cy = self.widget.bbox("insert")
            # Lower corner of the widget
            x += self.widget.winfo_rootx() + self.widget.winfo_width()
            y += self.widget.winfo_rooty() + self.widget.winfo_height()
            # creates a toplevel window
            self.tw = tk.Toplevel(self.widget)
            # Leaves only the label and removes the app window
            self.tw.wm_overrideredirect(True)
            self.tw.wm_geometry("+%d+%d" % (x, y))
            label = tk.Label(self.tw, text=self.text, justify='left',
                        background="#ffffff", relief='solid', borderwidth=1,
                        wraplength = self.wraplength)
            label.pack(ipadx=1)

        def hidetip(self):
            '''
            Hide the tooltip.
            '''
            tw = self.tw
            self.tw= None
            if tw:
                tw.destroy()

    def EntryCursorRight(main_frame, textvariable='', width=30, paddings=None, row=0, column=0):
        '''
        Create an entry with the cursor to the right.

        Args:
            main_frame: ttk.Frame, the main frame of the GUI
            textvariable: str, the text variable of the entry (default: '')
            width: int, the width of the entry  (default: 30)
            paddings: dict, the paddings of the entry (default: None)
            row: int, the row to place the entry (default: 0)
            column: int, the column to place the entry (default: 0)

        Returns:
            ttk.Entry, the entry
            '''
        entry = ttk.Entry(main_frame, textvariable=textvariable, width=width)
        entry.grid(**paddings, row=row, column=column)
        cursor_to_right(entry)
        return entry
    
    def update_widgets():
        '''
        Update the widgets based on the selected values.
        '''
        # Show/Hide the optional fields based on the selected values
        if video_demo_var.get():
            video_options.grid()
        else:
            video_options.grid_remove()
    
    def HelpButton(main_frame, image, helptext, paddings=None, row=0, column=0):
        '''
        Create a help button to display a tooltip when hovering over it.

        Args:
            main_frame: ttk.Frame, the main frame of the GUI
            image: ImageTk.PhotoImage, the image of the help button
            helptext: str, the help text of the button
            paddings: dict, the paddings of the button (default: None)
            row: int, the row to place the button (default: 0)
            column: int, the column to place the button (default: 0)
        '''
        help_button = ttk.Label(main_frame, image=image)
        help_button.grid(**paddings, row=row, column=column, sticky=tk.W)
        CreateToolTip(help_button, helptext)
        return help_button
    
    class label_entry_help_row():
        '''
        Create a label, an entry and a help button in a row.
        '''
        def __init__(self, main_frame, textvariable, paddings, main_label=None, validatecommand=None, helptext=None, help_icon=None, width=20):
            '''
            Initialize the label_entry_help_row class.

            Args:
                main_frame: ttk.Frame, the main frame of the GUI
                textvariable: dict, the text variables of the entry
                paddings: dict, the paddings of the entry
                main_label: str, the main label of the row (default: None)
                validatecommand: dict or function, the command to validate the entry (default: None)
                helptext: dict or str, the help text of the button (default: None)
                help_icon: ImageTk.PhotoImage, the image of the help button (default: None)
                width: int, the width of the entry (default: 20)
            '''
            # Label & Entry
            self.frame = ttk.Frame(main_frame)
            column = 0
            help_columns = []
            if main_label is not None:
                ttk.Label(self.frame, text=main_label).grid(**paddings, row=0, column=column, sticky=tk.W)
                column += 1
            for key, value in textvariable.items():
                # Get the command
                if isinstance(validatecommand, dict):
                    command= validatecommand[key] if key in validatecommand else None
                elif validatecommand is not None:
                    command = validatecommand
                else:
                    command = None
                ttk.Label(self.frame, text=key).grid(**paddings, row=0, column=column, sticky=tk.W)
                if isinstance(value, tk.BooleanVar):
                    ttk.Checkbutton(self.frame, variable=value, command=command).grid(**paddings, row=0, column=column+1)
                else:
                    ttk.Entry(self.frame, textvariable=value, validate='key', validatecommand=command, width=width).grid(**paddings, row=0, column=column+1)
                if isinstance(helptext, dict) and help_icon is not None and key in helptext:
                    HelpButton(self.frame, help_icon, helptext[key], paddings=paddings, row=0, column=column+2)
                    help_columns.append(column+2)
                    column += 3
                else:
                    column += 2
            # Help either for the whole row or for each entry
            if isinstance(helptext, str) and help_icon is not None:
                HelpButton(self.frame, help_icon, helptext, paddings=paddings, row=0, column=column)
                help_columns.append(column)
                column += 1

            self.frame.grid_rowconfigure(0, weight=1, minsize=30)
            self.frame.grid_columnconfigure([i for i in range(column) if i not in help_columns], weight=1, minsize=30)
        
        def grid(self, **kwargs):
            '''
            Grid the frame.
            '''
            self.frame.grid(**kwargs)

        def grid_remove(self):
            '''
            Grid remove the frame.
            '''
            self.frame.grid_remove()

    def select_input(selected):
        '''
        Select an input folder using a folder dialog.
        '''
        input_video_var.set(selected)
        cursor_to_right(entry_input)
    
    def plot_trailer(trailer_imgs):
        '''
        Plot the trailer in a Matplotlib figure.

        Args:
            trailer_imgs: list, N images
        
        Returns:
            Figure, the Matplotlib figure
        '''
        # Get the size to control figsize
        lines = trailer_imgs[0].shape[0]
        cols = trailer_imgs[0].shape[1]
        target_height = 2
        fig = plt.figure(figsize=(len(trailer_imgs)*target_height*cols/lines, target_height), dpi=100)
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        for i, img in enumerate(trailer_imgs):
            ax = fig.add_subplot(1, len(trailer_imgs), i+1)
            ax.imshow(img)
            ax.axis('off')
            ax.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        # Tight layout
        fig.tight_layout()
        return fig

    def select_output():
        '''
        Select an output folder using a folder dialog.
        '''
        selected = filedialog.askdirectory()
        output_var.set(selected)
        cursor_to_right(entry_output)

    def cursor_to_right(entry):
        '''
        Move the cursor to the right of the entry.

        Args:
            entry: ttk.Entry, the entry
        '''
        # Move the cursor to the end of the text
        entry.icursor(tk.END)
        # Ensure the view is scrolled to the end
        entry.xview_moveto(1.0)

    def load_with_PIL(image_path):
        '''
        Load an image with PIL and transpose it if needed.

        Args:
            image_path: str, the path to the image
        
        Returns:
            Image, the loaded image
        '''
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    
    def create_image_fig(image_files, output_resolution):
        '''
        Create a Matplotlib figure with a mosaic of images.

        Args:
            image_files: list, the list of image file paths
            output_resolution: tuple, the output resolution (width, height)

        Returns:
            Figure, the Matplotlib figure
        '''
        # Load images
        images = [load_with_PIL(file) for file in image_files]
        
        # Calculate the number of rows and columns for the mosaic
        num_images = len(images)
        num_columns = math.ceil(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_columns)
        
        # Calculate the size of each cell in the mosaic
        mosaic_width, mosaic_height = output_resolution
        cell_width = mosaic_width // num_columns
        cell_height = mosaic_height // num_rows
        
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(mosaic_width / 100, mosaic_height / 100))
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        
        # Flatten axes array for easy iteration
        axes = axes.flatten()
        
        # Position each image in the mosaic
        for index, img in enumerate(images):
            # Resize image to fit within its cell while maintaining aspect ratio
            img.thumbnail((cell_width, cell_height), Image.LANCZOS)
            
            # Display the image on the corresponding subplot
            axes[index].imshow(img)
            axes[index].axis('off')  # Hide the axes
            axes[index].margins(0,0)  # Remove margins
        
        # Hide any unused subplots
        for ax in axes[num_images:]:
            ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def resize_figure(fig, max_width, max_height):
        '''
        Resize a figure while maintaining the aspect ratio.

        Args:
            fig: Figure, the figure to resize
            max_width: int, the max width target of the figure
            max_height: int, the max height target of the figure
        '''
        # Get the original size of the figure
        original_width, original_height = fig.get_size_inches()

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Determine the new size according to the target size
        if max_width / max_height > aspect_ratio:
            new_width = max_height * aspect_ratio
            new_height = max_height
        else:
            new_width = max_width
            new_height = max_width / aspect_ratio

        # Resize the figure
        fig.set_size_inches(new_width, new_height)

    def display_img(img_ori, img_ph, max_height=None, max_width=None, max_mosaic=9, extensions=('.png','.jpg','.jpeg','.bmp','.obj'), warning_message=None, update_only=False, colorbar=None, annotations=None, annotations_color=None):
        '''
        Display an image or Matplotlib figure in the GUI.

        Args:
            img_ori: str, np.ndarray, or Figure, the image or figure to display
            img_ph: ttk.Frame, the placeholder for the image
            max_height: int, the maximum height of the image
            max_width: int, the maximum width of the image
        '''
        # Get the maximum size of the image
        if max_height == 0 or max_height is None:
            max_height = 200
        if max_width == 0 or max_width is None:
            max_width = 200

        # Differentiate between image and figure
        if isinstance(img_ori, Figure):
            fig = img_ori
            # Resize the figure to fit within the placeholder
            resize_figure(fig, max_width/100, max_height/100)
        elif isinstance(img_ori, str) and os.path.isdir(img_ori):
            imgs = [os.path.join(img_ori, f) for f in os.listdir(img_ori) if os.path.isfile(os.path.join(img_ori, f)) and f.lower().endswith(extensions)]
            if len(imgs) > max_mosaic:
                imgs = random.sample(imgs, max_mosaic-1) + [etc_path]
            if len(imgs) > 0:
                fig = create_image_fig(imgs, (max_width, max_height))
            else:
                warning_message = "Warning: No images found in the folder. Supported extensions are %s." % (', '.join(extensions))
                display_img(help_path, img_ph, max_height=max_height, max_width=max_width, warning_message=warning_message)
                return
        else:
            if isinstance(img_ori, str):
                if os.path.isfile(img_ori) and img_ori.endswith(extensions):
                    img = load_with_PIL(img_ori)
                elif img_ori == '':
                    img = load_with_PIL(none_path)
                else:
                    warning_message = "Warning: Not a valid Image. Supported extensions are %s." % (', '.join(extensions))
                    img = load_with_PIL(help_path)
            else:
                # Check number of channels to convert to RGBA
                if len(img_ori.shape) == 3 and img_ori.shape[2] == 3:
                    img = cv2.cvtColor(img_ori, cv2.COLOR_RGB2RGBA)
                else:
                    img = img_ori
                # img = Image.fromarray(img)
            
            # Create a Matplotlib figure
            if isinstance(img, Image.Image):
                img_width, img_height = img.size
            else:
                img_width, img_height = img.shape[1], img.shape[0]

            resize_ratio = min(max_height/img_height, max_width/img_width)

            if update_only:
                # Use the same size as the previous figure
                fig = img_ph.canvas.figure
                # Remove the previous elements
                if hasattr(fig, 'im_elements'):
                    for el in fig.im_elements:
                        try:
                            el.remove()
                        except:
                            pass
                if hasattr(fig, 'im'):
                    fig.im.remove()
                fig.set_size_inches(img_width*resize_ratio/100, img_height*resize_ratio/100)
            else:
                fig = Figure(figsize=(img_width*resize_ratio/100, img_height*resize_ratio/100))
                fig.ax = fig.add_subplot(111)
                fig.ax.axis('off')  # Hide axes
                fig.im_elements = []
                ## Set the limits of the image
                fig.ax.set_xlim(0, img_width)
                fig.ax.set_ylim(img_height, 0)
                fig.ax.margins(0,0)  # Remove margins

            if colorbar is not None:
                # Display the image with a colorbar
                fig.im = fig.ax.imshow(img, cmap=colorbar['cmap'], vmin=colorbar['vmin'], vmax=colorbar['vmax'])
                # Add a colorbar
                fig.im_elements.append(fig.colorbar(fig.im, ax=fig.ax))
            else:
                # Display the image without a colorbar
                fig.im = fig.ax.imshow(img)
            # Draw annotations if provided
            if annotations is not None:
                for annotation_key in annotations.keys():
                    fig.im_elements.extend(
                            draw_annotation(
                                fig.ax,
                                annotations[annotation_key],
                                annotation_key,
                                color= annotations_color[annotation_key] if annotations_color is not None and annotation_key in annotations_color else 'white',
                                fontsize=8,
                                linewidth=1
                            )
                        )
            
        # Set the background of the figure to be transparent
        fig.patch.set_facecolor((0,0,0,0))
        # Tight layout
        fig.tight_layout()
        # No padding
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
         
        # Update the canvas or create a new one
        if not hasattr(img_ph, 'canvas'):
            # Initialize the canvas with a figure of the same size than the target figure - to avoid wrong sizing with the first update
            img_ph.canvas = FigureCanvasTkAgg(Figure(figsize=(fig.get_size_inches()[0], fig.get_size_inches()[1]), facecolor=(0,0,0,0)), master=img_ph)
            img_ph.canvas.get_tk_widget().configure(background=bg)            
            img_ph.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, anchor='center')
            img_ph.canvas.get_tk_widget().update()
            
        # Close and Clear the previous figure
        if not update_only:
            plt.close(img_ph.canvas.figure)
            img_ph.canvas.figure.clear()
            img_ph.canvas.draw()
            img_ph.canvas.figure = fig
        # Update the size of the canvas
        img_ph.canvas.get_tk_widget().configure(width=fig.get_size_inches()[0]*100, height=fig.get_size_inches()[1]*100)
        img_ph.canvas.draw()
        img_ph.canvas.get_tk_widget().update()

        # Display the warning message if any
        if warning_message is not None:
            messagebox.showwarning("Warning", warning_message, parent=root)

    def display_img_in_gui(img_ori, text=None, max_height=max_height_gui, max_width=max_width_gui, update_only=False, colorbar=None, annotations=None, annotations_color=None, async_=False, skip_if_running=True):
        '''
        Display an image in the GUI.

        Args:
            img_ori: str or np.ndarray, the image to display
            text: str, the text to display
            max_height: int, the maximum height of the image
            max_width: int, the maximum width of the image
            update_only: bool, if True, the image is updated in place
            colorbar: dict, the colorbar parameters
            annotations: dict, the annotations to display
            annotations_color: dict, the colors of the annotations
            async_: bool, if True, the image is displayed asynchronously
            skip_if_running: bool, if True, the image is not displayed if processing is ongoing
        '''
        global _display_future

        def _display():
            display_img(
                img_ori,
                output_ph,
                max_height=max_height,
                max_width=max_width,
                update_only=update_only,
                colorbar=colorbar,
                annotations=annotations,
                annotations_color=annotations_color
            )

            # Add text if provided
            if text is not None:
                img_text.config(text=text)
        
        if not async_:
            if _display_future is not None and not _display_future.done():
                try:
                    _display_future.result(timeout=SYNC_WAIT_TIMEOUT)
                except TimeoutError:
                    # Log and continue safely
                    if log:
                        log.warning("Previous async display still running; forcing sync display")
            _display()
            return
        
        # async mode
        if skip_if_running and _display_future is not None and not _display_future.done():
            return

        _display_future = _display_executor.submit(_display)


    def interrupt_processing():
        '''
        Button callback to interrupt the processing.
        '''
        global GUI_STOP_REQUESTED
        ttk_terminal.config(state=tk.NORMAL)
        ttk_terminal.insert(tk.END, 'Processing interruption requested... Please wait.\n')
        ttk_terminal.see(tk.END)
        ttk_terminal.config(state=tk.DISABLED)
        GUI_STOP_REQUESTED = True

    def start_processing():
        '''
        Button callback to start the processing.
        '''
        global GUI_PROCESSING, GUI_STOP_REQUESTED
        if GUI_PROCESSING:
            messagebox.showinfo("Processing", "Processing is already started. Press interrupt to stop the processing.")
        else:
            GUI_STOP_REQUESTED = False
            threading.Thread(target=start_thread).start()
    
    def disable_buttons(disable):
        '''
        Disable or enable the GUI critical buttons and update the GUI.

        Args:
            state: str, the state of the buttons
        '''
        if disable:
            # Disable quit and start buttons
            start_button.config(state=tk.DISABLED)
            quit_button.config(state=tk.DISABLED)
            root.protocol("WM_DELETE_WINDOW", disable_close)
        else:
            # Enable quit and start buttons
            start_button.config(state=tk.NORMAL)
            quit_button.config(state=tk.NORMAL)
            root.protocol("WM_DELETE_WINDOW", root.destroy)

    def disable_close():
        '''
        Disable the close button of the window of the GUI
        '''
        messagebox.showinfo("Info", "Close button disabled")

    def start_thread():
        '''
        Start the processing in a separate thread to avoid blocking the GUI.
        Still, no other thread can be started until the current one is finished.
        '''
        global GUI_PROCESSING
        GUI_PROCESSING = True
        # Update args with values from GUI
        args.input_video = input_video_var.get()
        args.output = output_var.get()
        args.max_res = max_res_var.get()
        args.video_demo = video_demo_var.get()
        args.det_score = det_score_var.get()
        args.min_calib = min_calib_var.get()
        args.max_calib = max_calib_var.get()
        args.keep_tmp = keep_tmp_var.get()
        args.del_imgs = del_imgs_var.get()
        args.display_fct = display_img_in_gui

        # Infer some arguments
        if check_args_fct is not None:
            check_args_fct(args)
        
        log_file = os.path.join(args.output, '%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
        log = AppLog(ttk_terminal, terminal_size=terminal_size, log_file=log_file)

        print_and_log("Processing started with the selected parameters.", log=log)
        print_and_log("\tInput Folder: %s" % (args.input_video), log=log)
        print_and_log("\tOutput Folder: %s" % (args.output), log=log)
        print_and_log("\tDetection Score: %s %s second pass" % (args.det_score, "with" if args.rerun else "without"), log=log)
        if args.video_demo:
            print_and_log("\tVideo with max Resolution: %s and delete images: %s" % (args.max_res, args.del_imgs), log=log)
        else:
            print_and_log("No video demo selected.", log=log)
        print_and_log("\tKeep temporary files: %s" % (args.keep_tmp), log=log)
        
        # Start the processing
        try:
            disable_buttons(True)
            # Start processing
            main_function(args, log=log)
            print_and_log("Processing finished.", log=log)
        except Exception as e:
            # Trace the error
            trace_io = io.StringIO()
            traceback.print_exc(file=trace_io)
            print_and_log(trace_io.getvalue(), log=log)
            print_and_log("Processing stopped.", log=log) 
        close_log(log)
        GUI_PROCESSING = False
        disable_buttons(False)

    def select_font(preferred_font="ChollaSansOT"):
        '''
        Select the font to use in the GUI.
        '''
        from tkinter import font
        # list all available font families on the system
        available = font.families()        

        # fallback fonts (in order of preference)
        fallback_fonts = [
            "Segoe UI",     # Windows
            "Helvetica",    # macOS
            "Arial",        # universal safe fallback
            "Sans"          # Linux generic fallback
        ]

        # determine which font to use
        if preferred_font in available:
            chosen_font = preferred_font
        else:
            chosen_font = next((f for f in fallback_fonts if f in available), "Arial")

        return chosen_font

    def quit_app():
        '''
        Quit the application properly.
        '''
        # Make the window topmost to ensure the user sees the message box
        root.attributes('-topmost', True)
        if messagebox.askokcancel("Quit", "Do you really want to quit?", icon='warning'):
            global closed
            closed = True
            root.quit()
        root.attributes('-topmost', False)

    def disable_app():
        '''
        Disable the application.
        '''
        for widget in root.winfo_children():
            try:
                widget.configure(state='disabled')
            except tk.TclError:
                pass
    
    def enable_app():
        '''
        Enable the application.
        '''
        for widget in root.winfo_children():
            try:
                widget.configure(state='normal')
            except tk.TclError:
                pass

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.title("Physio-TIP")
    global closed
    closed = False
    log = None
    visuals_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'visuals')
    logo_path = os.path.join(visuals_path, 'ccp.png')
    etc_path = os.path.join(visuals_path, 'etc.png')
    none_path = os.path.join(visuals_path, 'none.png')
    help_path = os.path.join(visuals_path, 'help.png')

    # Theme and style for the GUI for rendering
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)
    style = ttk.Style()
    chosen_font = select_font(preferred_font="ChollaSansOT")
    style.theme_use("clam")        # clam / alt / default / classic
    style.configure("TButton",
                    font=(chosen_font, 11),
                    padding=3)
    style.configure("TLabel",
                    # background="#F4F6F7",
                    font=(chosen_font, 11))
    bg = style.lookup("TFrame", "background")

    # Define a custom style for your special button
    style.configure(
        "Big.TButton",
        font=(chosen_font, 12, "bold"),
        padding=(10, 6)  # optional, adjusts spacing nicely
    )

    # Help icon
    ## Resize the height to 20 and adjust the width accordingly to maintain aspect ratio
    help_icon = load_with_PIL(help_path)
    icon_width, icon_height = help_icon.size
    ratio = 20 / icon_height
    help_icon = help_icon.resize((int(icon_width * ratio), int(icon_height * ratio)))
    help_icon = ImageTk.PhotoImage(help_icon)

    # Layout
    paddings = {'padx': 3, 'pady': 2}

    # Variables
    input_video_var = tk.StringVar(value=args.input_video)
    output_var = tk.StringVar(value=args.output)
    max_res_var = tk.IntVar(value=args.max_res)
    video_demo_var = tk.BooleanVar(value=args.video_demo)
    det_score_var = tk.DoubleVar(value=args.det_score)
    min_calib_var = tk.DoubleVar(value=args.min_calib)
    max_calib_var = tk.DoubleVar(value=args.max_calib)
    keep_tmp_var = tk.BooleanVar(value=args.keep_tmp)
    del_imgs_var = tk.BooleanVar(value=args.del_imgs)

    # Validate commands
    val_com_float = (root.register(lambda x: x.replace('-','').replace('.','').isdigit() or x in ['','.','-']), '%P')
    val_com_float_pos = (root.register(lambda x: x.replace('.','').isdigit() or x in ['','.']), '%P')
    val_com_int_pos = (root.register(lambda x: x.isdigit() or x==''), '%P')
    val_com_float_pos_lw1 = (root.register(lambda x: (x.replace('.','').isdigit() and float(x) <= 1)  or x in ['','.']), '%P')

    width_entry = 40
    terminal_size = (90, 16)

    # Layout
    ## Logo
    logo_ph = ttk.Frame(main_frame)
    logo_ph.grid(**paddings, row=0, column=0, sticky=tk.NW)
    display_img(logo_path, logo_ph, max_height=max_height_logo, max_width=max_width_logo)

    ## Input/Output
    input_output_frame = ttk.Frame(main_frame)
    input_output_frame.grid(**paddings, row=1, column=0, sticky=tk.W)

    ttk.Label(input_output_frame, text="Video Input:").grid(**paddings, row=0, column=0, sticky=tk.W)
    entry_input = EntryCursorRight(input_output_frame, textvariable=input_video_var, width=width_entry, paddings=paddings, row=0, column=1)
    HelpButton(input_output_frame, help_icon, helptext_input_video, paddings=paddings, row=0, column=2)
    ttk.Button(input_output_frame, text="Browse Folder", command=lambda: select_input(filedialog.askdirectory())).grid(**paddings, row=0, column=3, sticky=tk.W)
    ttk.Button(input_output_frame, text="File", command=lambda: select_input(filedialog.askopenfilename())).grid(**paddings, row=0, column=4, sticky=tk.W)
    
    ttk.Label(input_output_frame, text="Output Folder:").grid(**paddings, row=1, column=0, sticky=tk.W)
    entry_output = EntryCursorRight(input_output_frame, textvariable=output_var, width=width_entry, paddings=paddings, row=1, column=1)
    HelpButton(input_output_frame, help_icon, helptext_output, paddings=paddings, row=1, column=2)
    ttk.Button(input_output_frame, text="Browse Folder", command=select_output).grid(**paddings, row=1, column=3, sticky=tk.W)

    input_output_frame.grid_rowconfigure([0, 1], weight=1, minsize=30)
    input_output_frame.grid_columnconfigure([0, 1, 2, 3, 4], weight=1, minsize=50)

    ## Detection Parameters
    det_param_frame = ttk.Frame(main_frame)
    det_param_frame.grid(**paddings, row=3, column=0, sticky=tk.W)
    ttk.Label(det_param_frame, text="Detection Parameters", font=('Arial', 12, 'bold')).grid(**paddings, row=0, column=0, columnspan=4, sticky=tk.W)

    label_entry_help_row(
        det_param_frame,
        textvariable={'min score': det_score_var},
        paddings=paddings,
        main_label="Detection:",
        validatecommand={'min score': val_com_float_pos_lw1},
        helptext={'min score': helptext_det_score},
        help_icon=help_icon,
        width=6,
    ).grid(**paddings, row=1, column=0, columnspan=4, sticky=tk.W)

    det_param_frame.grid_rowconfigure([0, 1], weight=1, minsize=30)
    det_param_frame.grid_columnconfigure([0, 1, 2, 3], weight=1, minsize=50)
        
    ## Visualization
    video_demo_frame = ttk.Frame(main_frame)
    video_demo_frame.grid(**paddings, row=5, column=0, sticky=tk.W)
    ttk.Label(video_demo_frame, text="Visualization", font=('Arial', 12, 'bold')).grid(**paddings, row=0, column=0, columnspan=2, sticky=tk.W)

    label_entry_help_row(
        video_demo_frame,
        textvariable={'Create Video:': video_demo_var},
        paddings=paddings,
        validatecommand={'Create Video:': update_widgets},
        helptext=helptext_video_demo,
        help_icon=help_icon,
        width=6,
    ).grid(**paddings, row=1, column=0, sticky=tk.W)

    video_options = label_entry_help_row(
        video_demo_frame,
        textvariable={'resolution': max_res_var, 'delete created images': del_imgs_var},
        paddings=paddings,
        validatecommand={'resolution': val_com_int_pos},
        helptext={'resolution': helptext_max_res, 'delete created images': helptext_del_imgs},
        help_icon=help_icon,
        width=6,
    )
    video_options.grid(**paddings, row=1, column=1, sticky=tk.W)

    video_demo_frame.grid_rowconfigure([0, 1], weight=1, minsize=30)
    video_demo_frame.grid_columnconfigure([0, 1], weight=1, minsize=50)

    ## Terminal
    terminal_frame = ttk.Frame(main_frame)
    terminal_frame.grid(**paddings, row=7, column=0, sticky=tk.W)
    # Decreased the font size from 10 to 8
    ttk_terminal = tk.Text(terminal_frame, width=terminal_size[0], height=terminal_size[1], state='disabled', font=('Courier New', 8), wrap=tk.WORD)
    log = AppLog(ttk_terminal, terminal_size=terminal_size)
    ttk_terminal.grid(**paddings, row=0, column=0, rowspan=3)

    # Center text on button
    start_button = ttk.Button(terminal_frame, text="Start", command=start_processing, style="Big.TButton")
    start_button.grid(**paddings, row=0, column=1)
    ttk.Button(terminal_frame, text="Stop", command=interrupt_processing, style="Big.TButton").grid(**paddings, row=1, column=1)
    quit_button = ttk.Button(terminal_frame, text="Quit", command=quit_app, style="Big.TButton")
    quit_button.grid(**paddings, row=2, column=1)

    terminal_frame.grid_rowconfigure([0, 1, 2], weight=1, minsize=30)
    terminal_frame.grid_columnconfigure([0, 1], weight=1, minsize=50)

    row_span = 8

    ## Placeholders for images and texts
    imgs_frame = ttk.Frame(main_frame)
    imgs_frame.grid(**paddings, row=0, column=1, rowspan=row_span, sticky=tk.NW)

    ### Input
    ttk.Label(imgs_frame, font='Helvetica 12 bold', text='Video Input').grid(**paddings, row=0, column=0, sticky=tk.S)
    input_ph = ttk.Frame(imgs_frame)
    input_ph.grid(**paddings, row=1, column=0)
    def input_var_callback(*args, previous_path_container=[''], extensions=('.mp4','.mkv','.avi','.mov', '.wmv', '.zip')):
        '''
        Callback when the input variable is changed.

        Args:
            previous_path_container: list, a list containing the previous path (to allow modification in the nested function)
            extensions: tuple, the supported video extensions
        '''
        img_path = input_video_var.get()
        if img_path != previous_path_container[0]:
            if os.path.exists(input_video_var.get()):
                disable_app()
                try:
                    print_and_log('Loading input video: %s\n\tThis may take some time (10s max)...' % (img_path), log=log)
                    trailer_imgs = get_trailer(img_path)
                    fig = plot_trailer(trailer_imgs)
                    display_img(fig, input_ph, max_height=max_height_input, max_width=max_width_input)
                    print_and_log('Input video loaded successfully.', log=log)
                except Exception as e:
                    display_img(help_path, input_ph)
                    # Warning if a trailer cannot be retrieved
                    disable_app()
                    messagebox.showwarning("Warning", "Data not supported or not found. Please select another file/folder.")
                    enable_app()
                enable_app()
            else:
                display_img(none_path, input_ph)
            previous_path_container[0] = img_path
    input_video_var.trace_add('write', input_var_callback)
    input_var_callback()

    ### Output
    img_text = ttk.Label(imgs_frame, font='Helvetica 12 bold')
    img_text.grid(**paddings, row=2, column=0)

    output_ph = ttk.Frame(imgs_frame)
    output_ph.grid(**paddings, row=3, column=0)

    imgs_frame.grid_rowconfigure([0, 2], weight=1, minsize=30)
    imgs_frame.grid_rowconfigure(1, weight=5, minsize=80)
    imgs_frame.grid_rowconfigure(3, weight=5, minsize=200)
    imgs_frame.grid_columnconfigure(0, weight=5, minsize=200)

    # Update the GUI
    update_widgets()

    # Configure the grid to expand properly
    main_frame.grid_rowconfigure(list(np.arange(0,row_span)), weight=1, minsize=30)
    main_frame.grid_columnconfigure([0, 1], weight=1, minsize=100)

    root.mainloop()

    # Closing app properly
    if closed: # Window is being closed
        root.destroy()
    elif log is not None: # Window was closed by the user
        log.close()
    return 1

def check_gui_stop(log=None):
    '''
    Check if the GUI requested to stop the process.

    Args:
        log: AppLog, the log to print the message (default: None)

    Returns:
        int, 1 if the process should be stopped, 0 otherwise
    '''
    if GUI_STOP_REQUESTED:
        print_and_log('Process stopped.', log=log)
        return 1
    return 0