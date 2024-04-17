from inspect import getframeinfo, stack

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[39m"
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

INFO    = BRIGHT_BLUE   + "[INFO]"    + RESET
INFO_   =                 "      "
WARNING = BRIGHT_YELLOW + "[WARNING]" + RESET
WARNING_=                 "         "
ERROR   = BRIGHT_RED    + "[ERROR]"   + RESET
ERROR_  =                 "       "
DEBUG   = BRIGHT_CYAN   + "[DEBUG]"   + RESET
DEBUG_  =                 "       "


__hide_debug__ = False
def hide_debug():
    global __hide_debug__
    __hide_debug__ = True
__hide_warning__ = False
def hide_warning():
    global __hide_warning__
    __hide_warning__ = True
__hide_info__ = False
def hide_info():
    global __hide_info__
    __hide_info__ = True

__find_caller__ = False
def find_caller():
    global __find_caller__
    __find_caller__ = True




def is_color(color):
    return (color[:2] == '\x1b[' and len(color) == 5)

def prntC(*values, sep=' ', end=RESET+'\n', start=RESET, flush=False):
    values = [v.__str__() for v in values]

    if (__hide_info__ and values[0] == INFO): return
    if (__hide_warning__ and values[0] == WARNING): return
    if (__hide_debug__ and values[0] == DEBUG): return
    
    string = start
    for i in range(len(values)):
        string += values[i]
        if i + 1 < len(values) and not(is_color(values[i])): # and not(is_color(values[i + 1])):
            string += sep
    print(string, end=end, flush=flush)

    if (__find_caller__):
        caller = getframeinfo(stack()[1][0])
        print(f"{BRIGHT_BLACK}Called from {caller.filename} at line {caller.lineno}{RESET}", flush=flush)

