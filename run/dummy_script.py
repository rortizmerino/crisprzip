import sys


def main(out_path='results/', array_id=1):
    # do cool stuff
    pass


if __name__ == "__main__":

    # (cluster) keyword arguments: array_id and out_path
    kwargs = {}
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
