from inspect import getfullargspec, getmembers
from typing import Tuple

def get_arguments(member):
    return [x for x in getfullargspec(member).args if x != "self"]

def get_optional_arguments(member):
    return (getfullargspec(member).kwonlydefaults or {}).items()

def get_arg_type(member, arg):
    anno = getfullargspec(member).annotations
    if arg not in anno:
        return None
    else:
        if isinstance(anno[arg], Tuple):
            return anno[arg][0]
        else:
            return anno[arg]


def get_arg_description(member, arg):
    anno = getfullargspec(member).annotations
    if arg not in anno:
        return ""
    else:
        if isinstance(anno[arg], Tuple):
            return anno[arg][1]
        elif isinstance(anno[arg], str):
            return anno[arg]
        else:
            return ""

def get_available_commands(cls):
    return [name for name, _ in getmembers(cls, callable) if not name.startswith("_")]