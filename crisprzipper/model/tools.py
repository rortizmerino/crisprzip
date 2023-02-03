"""
This module is a collection of some general tools that come in
handy in different modules.

Functions:
    path_handling()

"""

from inspect import signature
from pathlib import Path


def path_handling(func):
    """This decorator can be applied to any function taking on an
    argument called "path", and transforms strings to Path objects."""

    # finds all arguments that have pathlib.Path as (partial) signature
    params = signature(func).parameters
    path_args = [
        p for p in params
        if str(params[p].annotation).find("pathlib.Path") > -1
    ]

    def handled_func(*args, **kwargs):
        bnd_args = signature(func).bind(*args, **kwargs)
        bnd_args.apply_defaults()
        for path_arg in path_args:
            raw_path = bnd_args.arguments[path_arg]
            # replaces raw paths by Paths where necessary
            if not isinstance(raw_path, Path) and raw_path is not None:
                bnd_args.arguments[path_arg] = Path(raw_path)
        return func(*bnd_args.args, **bnd_args.kwargs)

    return handled_func
