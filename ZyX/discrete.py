"""Discrete Field maps"""
from numpy import mean, diff, load, empty, arange, pi, zeros, abs, logical_and, NaN, eye, log, array, unique, copy, moveaxis, polyfit, select
from numpy.linalg import solve, cond, lstsq
from scipy.interpolate import RectBivariateSpline

from .basic import AbstractField

def rectBS(z, x, zPlane, xPlane, bY, dZ, dX, maxord=4, s=0):
    rbs = RectBivariateSpline(zPlane, xPlane, bY, kx=maxord, ky=maxord, s=s)
    return rbs(z,x, dZ, dX, grid=False)


class SplineMap(AbstractField):
    """one field component in two dimensions"""
    def __init__(self, zPlane, xPlane, fld, flipX=False, maxord: int=4, mirrorZ: bool=True, mm=False):
        assert zPlane.size == fld.shape[0]
        assert xPlane.size == fld.shape[1]
        self.zPlane = zPlane/1e3 if mm else zPlane
        self.xPlane = xPlane/1e3 if mm else xPlane
        self.fld = copy(fld) # safety measures, as 3d array might get scaled when scaling this slice later
        print(f'shapes: z={zPlane.size}, x={xPlane.size}, fld={fld.shape}')
        self.zLim = array([self.zPlane[0], self.zPlane[-1]])
        self.xLim = array([self.xPlane[0], self.xPlane[-1]])
        self.maxord = maxord
        AbstractField.__init__(self, normalized=False)
        self.transform.mirrorZ = mirrorZ
        self.smoothingFactor=0  # 0 means strict interpolation

    def scale_fields(self, factor):
        self.fld *= factor

    def _normalize(self, rigidity):
        self.scale_fields(1 / rigidity)

    def _curvature(self, z, x):
        zTrans, xTrans = self.transform.map(z, x)
        return rectBS(zTrans, xTrans, self.zPlane, self.xPlane, self.fld, 0, 0, self.maxord, self.smoothingFactor)

    def onPath(self, z, x):
        return select((logical_and(z < self.zLim[1], x > self.xLim[0]),),
                         (self._curvature(z, x),))

def in_ival(x, xLim):
    return logical_and(x >= xLim[0], x <= xLim[1])

def gridFromList(zList,xList,bList):
    Z, iz = unique(zList, return_inverse=True)
    X, ix = unique(xList, return_inverse=True)
    B=empty((len(Z), len(X)))
    B.fill(NaN)
    B[iz,ix] = bList
    return Z, X, B
