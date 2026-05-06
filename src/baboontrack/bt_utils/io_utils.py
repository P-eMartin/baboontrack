
import os
import logging
import numpy as np
import platform
import sys
import datetime
import subprocess
import csv
import zipfile

'''
Print and log functions
'''
def print_and_log(message, log=None, to_print=True):
    '''
    Print and log a message.
    
    Args:
        message: str, the message to print and log
        log: logging.Logger, the logger to log the information (facultative, default None)
        to_print: bool, to print the message (default True)
    '''
    if to_print:
        # To delete the content of the current line
        sys.stdout.write('\x1b[2K')
        print(message)
    if log is not None:
        log.info(message)


def setup_logger(logger_name='my_log', log_file='%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")), level=logging.INFO):
    '''
    Setup a logger to log information.
    
    Args:
        logger_name: str, the name of the logger (default 'my_log')
        log_file: str, the path to the log file (default 'my_log.log')
        level: int, the level of the logger (default logging.INFO)
    '''
    if not os.path.isdir(os.path.dirname(os.path.realpath(log_file))):
        os.makedirs(os.path.dirname(os.path.realpath(log_file)))
    l = logging.getLogger(logger_name)
    # Avoid double printing in the terminal if the logger is already set up
    l.propagate = False
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    return l

def close_log(log):
    '''
    Close the log file.
    
    Args:
        log: logging.Logger, the logger to log the information
    '''
    if log is not None:
        x = list(log.handlers)
        for i in x:
            log.removeHandler(i)
            i.flush()
            i.close()

def progress_bar(count, total, title, completed=0, log=None):
    '''
    Print a progression bar in the terminal. If completed is set to 0, the progression bar will be updated.

    Args:
        count: int, the current count
        total: int, the total count
        title: str, the title of the progression bar
        completed: int, the completed status of the progression bar (default 0) to print definitively
        log: logging.Logger, the logger to log the information (default None)
    '''
    def get_terminal_size():
        '''
        Get the terminal size for different platform.

        Returns:
            tuple: the terminal size
        '''
        def _get_terminal_size_windows():
            try:
                from ctypes import windll, create_string_buffer
                import struct
                h = windll.kernel32.GetStdHandle(-12)
                csbi = create_string_buffer(22)
                res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
                if res:
                    (bufx, bufy, curx, cury, wattr,
                    left, top, right, bottom,
                    maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                    sizex = right - left + 1
                    sizey = bottom - top + 1
                    return sizex, sizey
            except:
                pass

        def _get_terminal_size_tput():
            try:
                import shlex
                cols = int(subprocess.check_call(shlex.split('tput cols')))
                rows = int(subprocess.check_call(shlex.split('tput lines')))
                return (cols, rows)
            except:
                pass

        def _get_terminal_size_linux():
            def ioctl_GWINSZ(fd):
                try:
                    import fcntl, termios, struct
                    cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
                    return cr
                except:
                    pass
            cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
            if not cr:
                try:
                    fd = os.open(os.ctermid(), os.O_RDONLY)
                    cr = ioctl_GWINSZ(fd)
                    os.close(fd)
                except:
                    pass
            if not cr:
                try:
                    cr = (os.environ['LINES'], os.environ['COLUMNS'])
                except:
                    return None
            return int(cr[1]), int(cr[0])

        current_os = platform.system()
        tuple_xy = None
        if current_os == 'Windows':
            tuple_xy = _get_terminal_size_windows()
            if tuple_xy is None:
                tuple_xy = _get_terminal_size_tput()
                # needed for window's python in cygwin's xterm!
        if tuple_xy is None and (current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN')):
            tuple_xy = _get_terminal_size_linux()
        if tuple_xy is None:
            tuple_xy = (80, 25)      # default value
        return tuple_xy
    if log is not None and hasattr(log, 'terminal_size'):
        terminal_size = log.terminal_size
    else:
        terminal_size = get_terminal_size()
    percentage = int(100.0 * count / total)
    length_bar = min([max([3, terminal_size[0] - len(title) - len(str(total)) - len(str(count)) - len(str(percentage)) - 10]),20])
    filled_len = int(length_bar * count / total)
    bar = '█' * filled_len + ' ' * (length_bar - filled_len)
    # To delete the content of the current line
    sys.stdout.write('\x1b[2K')
    sys.stdout.write('%s [%s] %s %% (%d/%d)\r' % (title, bar, percentage, count, total))
    sys.stdout.flush()
    if completed:
        print_and_log('%s [%s] %s %% (%d/%d)' % (title, bar, percentage, count, total), log)
    elif log is not None:
        log.info('%s [%s] %s %% (%d/%d)' % (title, bar, percentage, count, total))

'''
Filesystem functions
'''
def save_dict_as_csv(dict_to_save, save_path, extra_fields_before=None, extra_fields_after=None):
    '''
    Save a dictionary as a csv file.

    Args:
        dict_to_save: dict, the dictionary to save. Keys are the headers and values are a list of values.
        save_path: str, the path to save the csv file
        extra_fields_before: dict, the extra fields to add before the dictionary keys (default None)
        extra_fields_after: dict, the extra fields to add after the dictionary keys (default None)
    '''
    extra_fields_before = extra_fields_before or {}
    extra_fields_after = extra_fields_after or {}

    headers = list(extra_fields_before.keys()) + list(dict_to_save.keys()) + list(extra_fields_after.keys())
    all_fields = {**extra_fields_before, **dict_to_save, **extra_fields_after}
    num_rows = max(len(values) for values in all_fields.values())

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for i in range(num_rows):
            row = [(all_fields[key][i] if i < len(all_fields[key]) else '') for key in headers]
            writer.writerow(row)

def zip_folder(folder_to_zip, output_file):
    '''
    Zip a folder.
    Args:
        folder_to_zip: str, the path to the folder to zip
        output_file: str, the path to save the zip file
    
    Returns:
        int, 1 if the folder was properly zipped
    '''
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.dirname(output_file)))
    return 1

'''
Miscellaneous functions
'''
def get_value_with_precision(value, precision=1000):
    '''
    Get the value with a certain precision.
    Useful for saving in json files (or other) to avoid float precision issues.
    
    Args:
        value: float, the value to process
        precision: int, the precision to use (default 1000)
    
    Returns:
        float: the value with the precision
    '''
    if type(value) is list:
        return np.trunc(precision*np.array(value))/precision
    return np.trunc(precision*value)/precision

