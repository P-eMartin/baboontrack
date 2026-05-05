import json
import ast
import numpy as np

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

def save_json_file(variable, json_file, cls=NpEncoder):
    '''
    Save json files with a customized encoder.

    Args:
        variable: variable, the variable to save in the json file
        json_file: str, path to the json file
        cls: class, the class to use for the encoder (default NpEncoder)

    Returns:
        int, 1 if the file was properly saved
    '''
    with open(json_file, 'w') as f:
        json.dump(variable, f, cls=cls)
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
    with open(json_file, 'r') as f:
        variable=json.load(f)
    if type(variable) is not str or str_ok:
        return variable
    else:
        return ast.literal_eval(variable)