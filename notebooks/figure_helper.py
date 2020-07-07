import os
import time


path_figures = os.path.join(os.path.split(__file__)[0], 'figures')


class Figure(object):
    
    def __init__(self, path, fn="figure", suffix=".pdf"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.fn = fn
        self.suffix = suffix
        self.count = {}
    
    def save(self, fig, fn=None):
        if fn is None:
            fn = self.fn
        count = self.count.setdefault(fn, 0)
        fig.savefig(os.path.join(self.path, "{0}_{1}{2}".format(fn, count, self.suffix)))
        self.count[fn] = count + 1

class FigureHelper(object):
    
    def __init__(self):
        self._time = time.localtime()
        self._datestr = "{0}-{1}-{2}".format(self._time.tm_year, self._time.tm_mon, self._time.tm_mday)
    
    def __call__(self, figname):
        a_path = os.path.join(path_figures, figname, self._datestr)
        return Figure(a_path)
