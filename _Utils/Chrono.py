
import time

class Chrono:

    def __init__(self, start=True) -> None:
        self.time = time.time()
        self.stopped = False
        self.__elapsed_stop__ = 0

        if not(start):
            self.stop()

# |====================================================================================================================
# | ACTIONS
# |====================================================================================================================

    def reset(self) -> None:
        self.time = time.time()
        self.__elapsed_stop__ = 0

    def stop(self) -> float:
        self.__elapsed_stop__ = time.time() - self.time
        self.stopped = True

    def start(self) -> None:
        if (not self.stopped):
            self.reset()
            return

        self.time = time.time() - self.__elapsed_stop__
        self.__elapsed_stop__ = 0
        self.stopped = False

    def toggle(self) -> None:
        if (self.stopped):
            self.start()
        else:
            self.stop()

    def set_elapsed_time(self, elapsed:float) -> None:
        self.time = time.time() - elapsed
        if (self.stopped):
            self.__elapsed_stop__ = elapsed

# |====================================================================================================================
# | GETTERS
# |====================================================================================================================

    def is_stopped(self) -> bool:
        return self.stopped

    def get_time_s(self) -> float:
        return time.time() - self.time

    def get_time_m(self) -> float:
        return (time.time() - self.time) / 60

    def get_time_h(self) -> float:
        return (time.time() - self.time) / 3600

    def get_time_d(self) -> float:
        return (time.time() - self.time) / 86400


# |====================================================================================================================
# | OPERATORS OVERLOADING
# |====================================================================================================================

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:

        s, m, h, d = self.get_time_s(), 0, 0, 0
        ms = s - int(s)
        s = int(s)

        if (s >= 60):
            m = s // 60
            s %= 60
        if (m >= 60):
            h = m // 60
            m %= 60
        if (h >= 24):
            d = h // 24
            h %= 24

        if (d > 0):
            return f"{d}d {h}h {m}m {s}s"
        if (h > 0):
            return f"{h}h {m}m {s}s"
        if (m > 0):
            return f"{m}m {s}s"
        return f"{s}s"

