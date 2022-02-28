import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import numpy as np

from model.fit_analysis import LogAnalyzer, DashboardVideo

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

    out_path = os.path.join(out_dir, 'GSA_20220215.mp4')
    log_list = [os.path.join(root_dir,
                             f'run/vid_data/20220215_471745/{i:03d}/c_log.txt')
                for i in range(1, 11)]
    videomaker = DashboardVideo(log_list)
    videomaker.make_video(fps=150)
    videomaker.save_video(out_path, fps=150)


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
