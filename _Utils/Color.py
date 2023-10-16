
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

def is_color(color):
    return (color[:2] == '\x1b[' and len(color) == 5)

def prntC(*values, sep=' ', end=RESET+'\n', start=RESET):
    values = [v.__str__() for v in values]
    string = start
    for i in range(len(values)):
        string += values[i]
        if i + 1 < len(values) and not(is_color(values[i])): # and not(is_color(values[i + 1])):
            string += sep
    print(string, end=end)

