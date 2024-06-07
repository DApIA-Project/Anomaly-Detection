

import os
import sys


WINDOWS = (sys.platform == "win32")

if (WINDOWS):

# |--------------------------------------------------------------------------------------------------------------------
# | abspath wrapper
# |--------------------------------------------------------------------------------------------------------------------

    os_path_abspath = os.path.abspath
    def __abspath(path:str) -> str:
        return os_path_abspath(path).replace("/", "\\")
    os.path.abspath = __abspath

