import autograd.numpy as ag
from numpy import abs, ones
from numpy.polynomial.polynomial import polyval
from scipy.special import gamma

from .basic import AbstractField


class ZXfunMagnet(AbstractField):
    def __init__(self, length, polyCoefs: iter):
        AbstractField.__init__(self, normalized=True)
        self.length = length
        self.polyCoef = polyCoefs

    def normalize(self, rigidity):
        pass

    def _curvature(self, z, x):
        # overwrite by descendant classes
        pass

    def inPlane(self, z, x):
        zo, xo = self.transform.map(z.reshape([-1, 1]), x.reshape([1, -1]))
        return self._curvature(zo, xo)

    def onPath(self, z, x):
        zo, xo = self.transform.map(z.reshape([-1, 1]), x.reshape([-1, 1]))
        return self._curvature(zo, xo)[:, 0]


class GammaEdgeMagnet(ZXfunMagnet):
    def __init__(self, length, polyCoefs: iter, gammaN=30):
        ZXfunMagnet.__init__(self, length, polyCoefs)
        self._gammaN = gammaN
        self._prefac = 1 / gamma((gammaN + 1) / gammaN)

    def _curvature(self, z, x):
        return self._prefac * ag.exp(-(z / self.length) ** self._gammaN) * polyval(x, self.polyCoef)


class DemoSextupole(AbstractField):
    """for checking the potential derivatives. expression from Wille book"""
    def __init__(self):
        AbstractField.__init__(self, normalized=True)

    def potential(self, x, y, z):
        zTrans, xTrans = self.transform.map(z, x)
        return y*x**2 / 2 - y**3 / 6

    def potentialOnPath(self, x, y, z):
        return self.potential(x,y,z)

    def normalize(self, factor):
        pass

    def _curvature(self, z, x):
        return x**2

    def inPlane(self, z, x):
        zo, xo = self.transform.map(z.reshape([-1, 1]), x.reshape([1, -1]))
        return self._curvature(zo, xo)

    def onPath(self, z, x):
        zo, xo = self.transform.map(z.reshape([-1, 1]), x.reshape([-1, 1]))
        return self._curvature(zo, xo)[:, 0]


class LinearEdgeMagnet(ZXfunMagnet):
    def __init__(self, length: float, polyCoefs: iter, edgeLength: float):
        ZXfunMagnet.__init__(self, length, polyCoefs)
        self.linPart = edgeLength / 2

    def _curvature(self, z, x):
        zm = (abs(z) - self.length) / self.linPart
        g = ag.select(condlist=(zm < -1, zm < 1), choicelist=(ones(zm.shape), (1 - zm) / 2), default=0)
        return g * polyval(x, self.polyCoef)
