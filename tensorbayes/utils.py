import itertools
import numpy as np
import os
import sys

def _valid_type(arg_type):
    """Check if argument can be passed safely to list() or tuple()"""
    valid_types = [tuple, list, itertools.chain]
    return any([arg_type is t for t in valid_types])

def tuplify(arg):
    """Converts an iterable into a tuple"""
    if _valid_type(type(arg)):
        return tuple(arg)
    else:
        return (arg,)

def listify(arg, recursive=False):
    """Converts an iterable into a list"""
    if _valid_type(type(arg)):
        if recursive:
            arg = [listify(v, True) for v in arg]
            return list(itertools.chain.from_iterable(arg))
        else:
            return list(arg)
    else:
        return [arg]

def recursive_listify(arg):
    """"""
    if not _valid_type(type(arg)):
        return [arg]
    else:
        arg = [flatten_to_list(v) for v in arg]
        return list(itertools.chain.from_iterable(arg))

def rename(newname):
    def decorator(f):
        if newname is not None:
            f.__name__ = newname
        return f
    return decorator

def ask(question):
    while True:
        ans = raw_input(question + ' (y/n): ')
        if ans in {'y', 'n'}:
            return ans == 'y'

class Session(object):
    def __init__(self, fh=sys.stdout, save_model_path=None):
        print "[Session] Begin session..."
        self.fh = fh
        self.save_model_path = save_model_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if ask('Do you wish to delete the log?'):
                if os.path.exists(self.fh.name):
                    os.remove(self.fh.name)
                    print "[Session] Log deleted"
                else:
                    print "[Session] Log does not exist"
            if self.save_model_path is not None:
                if ask('Do you wish to delete the model?'):
                    if os.path.exists(self.save_model_path):
                        os.remove(self.save_model_path)
                    else:
                        print "[Session] Model does not exist"
        if isinstance(exc_val, KeyboardInterrupt):
            quit()

def file_handle(path, name, version, get_seed=False,
                test_run=False, seed=None, overwrite=False):
    print
    folder = os.path.join(path, name, version)
    if test_run:
        print "This is a test run. Directories are created. Files are not."

    if os.path.exists(folder):
        print "Directory already exists:", folder
    else:
        print "Creating directory:", folder
        os.makedirs(folder)

    if overwrite:
        assert seed is not None, "Cannot overwrite file without seed info"

    seed = 1000 if seed is None else seed
    filename = name + '_' + version + '_seed={0:4d}.out'.format(int(seed))
    filepath = os.path.join(folder, filename)
    if not overwrite:
        while os.path.exists(filepath):
            seed += 1
            filename = name + '_' + version + '_seed={0:4d}.out'.format(int(seed))
            filepath = os.path.join(folder, filename)

    if test_run:
        print "Would have written to:", filepath
        fh = sys.stdout
    else:
        print "Writing to:", filepath
        fh = open(filepath, 'w', 0)
    print

    if get_seed:
        return fh, seed
    else:
        return fh
