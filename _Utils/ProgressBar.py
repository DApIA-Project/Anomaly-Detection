import time


def sec_to_time(sec:float)->str:
    sec = int(sec)
    s = sec % 60
    sec -= s
    if (sec == 0):
        return str(s)+"s"

    m = (sec % 3600) // 60
    sec -= m * 60
    if (sec == 0):
        return str(m).rjust(2, "0")+"m "+str(s).rjust(2, "0")+"s"

    h = sec // 3600
    return str(h).rjust(2, "0")+"h "+str(m).rjust(2, "0")+"m "+str(s).rjust(2, "0")+"s"


# |====================================================================================================================
# | Genrate pleasant display of a progress bar with easy parameters
# |====================================================================================================================

class ProgressBar:
    def __init__(self, min:float=0, max:float=100, width:int=20) -> None:
        self.min = min
        self.max = max
        self.width = width
        self._progress = 0
        self._startup = time.time()
        self._reprint = True

    def reset(self, min:float=None, max:float=None, width:int=None) -> None:
        if min is not None:self.min = min
        else: self.min = 0

        if max is not None:
            self.max = max

        if width is not None:
            self.width = width

        self._progress = 0
        self._startup = time.time()

    def disable_reprint(self) -> None:
        self._reprint = False



    def update(self, value:float=None, additional_text="") -> None:
        if (value is None):
            value = self._progress + 1
        additional_text = additional_text.strip()
        if (value == self.min):
            self._startup = time.time()
        elapsed = time.time() - self._startup

        self._progress = value
        percent = 100 * (value - self.min) / (self.max - self.min)

        remaining_percent = 100 - percent
        remaining_time = (elapsed * remaining_percent / percent) if percent > 0 else 0


        if(self._reprint):

            text = "\r["
            text += "=" * int(self.width * percent / 100)
            if (percent < 100):
                text += ">"
                text += " " * (self.width - int(self.width * percent / 100) - 1)
            else:
                text += "=" * (self.width - int(self.width * percent / 100))
            text += "] "+str(int(percent))+"%"
            if (self.min == 0):
                text += " ("+str(round(value,1))+"/"+str(round(self.max,1))+")"

            text += " - remain : "  + sec_to_time(remaining_time)
            text += " - elapsed : " + sec_to_time(elapsed       )
            if (additional_text != ""):
                text += " - "+additional_text
            text += " "*20

            if (percent >= 100):
                text += "\n"

            print(text, end="")



    def __call__(self, value:float) -> None:
        self.update(value)