
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

def color_print(*values, sep=' ', end=RESET+'\n'):
    string = ''
    for i in range(len(values)):
        v = values[i]
        # if v is a color
        str_rep = v.__str__()
        if (str_rep[:2] == '\033' and len(str_rep) == 5):
            string += str_rep
        else:
            string += str_rep
            if (i < len(values)-1):
                string += sep
    print(string, end=end)
