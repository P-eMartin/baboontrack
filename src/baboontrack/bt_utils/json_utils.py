import json
import ast
import os
import numpy as np
import traceback

class NpEncoder(json.JSONEncoder):
    '''
    Class to encode numpy arrays in json files.
    
    Usage example:
    json.dumps(variable, cls=NpEncoder)
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json_file(variable, json_file, cls=NpEncoder, pretty=False):
    '''
    Save json files with a customized encoder.

    Args:
        variable: variable, the variable to save in the json file
        json_file: str, path to the json file
        cls: class, the class to use for the encoder (default NpEncoder)
        pretty: bool, if True, save the json file with indentation (default False)

    Returns:
        int, 1 if the file was properly saved
    '''
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(variable, f, cls=cls, indent=2 if pretty else None)
    return 1

def load_json_file(json_file, str_ok=False):
    '''
    Load variable from json file.

    Args:
        json_file: str, path to the json file
        str_ok: bool, if True, load the json file as a string (default False)

    Returns:
        variable: variable, the variable loaded from the json file
    '''
    try:
        with open(json_file, 'r') as f:
            variable=json.load(f)
    # If error: print the error and stop the program
    except Exception as e:
        print(f"Error loading json file {json_file}: {e}")
        traceback.print_exc()
        raise e
        
    if type(variable) is not str or str_ok:
        return variable
    else:
        return ast.literal_eval(variable)
    
def convert_keys(obj):

    if isinstance(obj, dict):
        return {
            str(k): convert_keys(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):
        return [convert_keys(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_keys(v) for v in obj)

    else:
        return obj


def save_dict_to_txt(obj, filepath, indent=2):
    """
    Save an object's __dict__ to a text file.

    Parameters
    ----------
    obj : object
        Any Python object with a __dict__ attribute.
    filepath : str
        Output file path.
    indent : int
        JSON indentation level.
    """

    data = convert_keys(obj.__dict__)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
