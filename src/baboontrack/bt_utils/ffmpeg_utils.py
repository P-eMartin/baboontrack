import os
import pdb
import subprocess
from .io_utils import print_and_log
import traceback

def get_ffmpeg_codecs(log=None):
    '''
    Get the codecs available with ffmpeg.

    Args:
        log: logging.Logger, the logger to log the information (default None)
    '''
    command = ['ffmpeg', '-hide_banner', '-codecs']
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        codecs = []
        for line in output.split('\n'):
            if 'DEV' in line:
                codecs.append(line.split()[1])
        return codecs
    except subprocess.CalledProcessError as e:
        print_and_log('Error during ffmpeg codecs listing: %s' % (str(e)), log=log)
        print_and_log('Traceback: %s' % (traceback.format_exc()), log=log)
        return []
    except Exception as e:
        print_and_log('An unexpected error occurred during ffmpeg codecs listing: %s' % (str(e)), log=log)
        print_and_log('Traceback: %s' % (traceback.format_exc()), log=log)
        return []

def get_ffmpeg_codec(log=None):
    '''
    Get the codec to use with ffmpeg.

    Args:
        log: logging.Logger, the logger to log the information (default None)
    '''
    if os.system('ffmpeg -version') != 0:
        print_and_log('Warning: ffmpeg is not installed. Please install ffmpeg to create videos.', log=log)
        return ''
    else:
        # Check which codec is available
        codecs = get_ffmpeg_codecs(log=log)
        if len(codecs) == 0:
            print_and_log('No codec available. Please install ffmpeg with codecs to create videos.', log=log)
            return ''
        elif 'libx265' in codecs:
            return 'libx265'
        elif 'h265' in codecs:
            return 'h265'
        elif 'hevc' in codecs:
            return 'hevc'
        elif 'libx264' in codecs:
            return 'libx264'
        elif 'h264' in codecs:
            return 'h264'
        elif 'libxvid' in codecs:
            return 'libxvid'
        else:
            print_and_log('Codecs available: %s. Selecting the first one.' % (', '.join(codecs)), log=log)
            return codecs[0]

def check_svg_support():
    try:
        # Run the ffmpeg -version command
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        # Check if the output contains the necessary flags
        if '--enable-libfreetype' in result.stdout and '--enable-libsvg' in result.stdout:
            return True
        else:
            return False
    except FileNotFoundError:
        # ffmpeg is not installed
        return False

def run_command(command, log=None):
    """
    Run a command in the terminal and stream its output in real time.

    Args:
        command: list[str]
            Command to execute.

        log: logging.Logger, optional
            Logger used by print_and_log().

    Returns:
        int
            Return code of the process (0 on success).
    """

    try:
        print_and_log(f"Running command: {' '.join(command)}", log=log)
        pdb.set_trace()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()

            # Skip progress-bar style updates, empty lines and warnings.warn
            if '%' in line or line.strip() == '' or 'warnings.warn' in line or "FutureWarning" in line:
                print(line, end='\r', flush=True)
            else:
                print_and_log('\t' + line, log=log)
        return_code = process.wait()
        if return_code != 0:
            print_and_log(f"Command {' '.join(command)} failed with return code {return_code}", log=log)

        return return_code

    except Exception as e:
        print_and_log(f"Command {' '.join(command)}:", log=log)
        print_and_log(f"\tAn unexpected error occurred when running the command: {e}", log=log)
        print_and_log(f"Traceback:\n{traceback.format_exc()}",log=log)
        return 1

def create_video(input_folder, output_file, fps=30, sequence='%d.png', threads=1, codec='libx265', log=None, extra_args=[]):
    '''
    Create video from a folder of images.

    Args:
        input_folder: str, path to folder containing the images
        output_file: str, path to output video
        fps: int, fps of the video
        sequence: str, sequence/pattern of the images - use glob is None
        threads: int, number of threads to use (default 1 - no parallelization/ 0 - optimized number of threads)
    '''
    tmp_file = os.path.join(input_folder, 'ffmpeg_tmp.txt')
    # Check if more than one image in the folder
    if len(os.listdir(input_folder)) > 1 and codec != '':
        # Create output folder if do not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        import subprocess
        if sequence is None:
            # Use a tmp file to store the list of images on windows
            if os.name == 'nt':
                with open(tmp_file, 'w') as f:
                    for img in sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'))]):
                        f.write('file \'%s\'\n' % (img))
                command = ['ffmpeg', '-hide_banner', '-y', '-threads', str(threads), '-loglevel', 'error', '-r', str(fps), '-f', 'concat', '-safe', '0', '-i', tmp_file, '-codec', codec]
            else:
                command = ['ffmpeg', '-hide_banner', '-y', '-threads', str(threads), '-loglevel', 'error', '-r', str(fps), '-pattern_type', 'glob', '-i', os.path.join(input_folder,'*.png'), '-codec', codec]
        else:
            command = ['ffmpeg', '-hide_banner', '-y', '-threads', str(threads), '-loglevel', 'error', '-r', str(fps), '-i', os.path.join(input_folder, sequence), '-codec', codec]
        
        command += extra_args
        # Add output file
        command += [output_file]
        run_command(command, log=log)
    return 1