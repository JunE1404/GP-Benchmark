from misc.scaffolds import LogDetails
from typing import Any
from typing_extensions import Callable

def logger(details: LogDetails):
    # do stuff with log details
    pass

def getLogger()-> Callable[[Any], None]:
    return logger