class SliceEx(object):
    def __init__(self, start, stop, step=None):
        self.start = start
        self.stop = stop
        self.step = step
    def __getitem__(self, shift):
        return slice(self.start + shift, self.stop + shift, self.step)