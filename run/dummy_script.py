import sys
import os

from model import *


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(script_path='./fit_data_da.py', out_path='results/',
         array_id=1):

    # collecting arguments
    root_dir = get_root_dir(script_path)
    out_dir = os.path.abspath(out_path)

    # print something
    print('You did it!')


if __name__ == "__main__":

    # (cluster) keyword arguments: script_path, array_id and out_path
    kwargs = {'script_path': sys.argv[0]}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, val = arg.split('=')
            if key == 'array_id':
                kwargs[key] = int(val)
            else:
                kwargs[key] = val

    # arguments: anything needed for this script
    args = [arg for arg in sys.argv[1:] if not ('=' in arg)]

    main(*args, **kwargs)
