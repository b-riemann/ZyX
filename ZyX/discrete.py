"""Discrete Field maps"""
import autograd.numpy as ag
from autograd.extend import primitive, defvjp
from numpy import mean, diff, load, empty, arange, pi, zeros, abs, logical_and, NaN, eye, log, array, unique, copy, moveaxis, polyfit
from numpy.linalg import solve, cond, lstsq
from scipy.interpolate import RectBivariateSpline

from .basic import AbstractField

@primitive
def rectBS(z, x, zPlane, xPlane, bY, dZ, dX, maxord=4, s=0):
    rbs = RectBivariateSpline(zPlane, xPlane, bY, kx=maxord, ky=maxord, s=s)
    return rbs(z,x, dZ, dX, grid=False)

defvjp(rectBS,
       lambda ans, z, x, zPlane, xPlane, bY, dZ, dX, maxord, s: lambda g: g * rectBS(z, x, zPlane, xPlane, bY, dZ+1, dX, maxord, s),
       lambda ans, z, x, zPlane, xPlane, bY, dZ, dX, maxord, s: lambda g: g * rectBS(z, x, zPlane, xPlane, bY, dZ, dX+1, maxord, s))


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
        return ag.select((logical_and(z < self.zLim[1], x > self.xLim[0]),),
                         (self._curvature(z, x),))

def in_ival(x, xLim):
    return logical_and(x >= xLim[0], x <= xLim[1])

class YExtendedSplineMap(SplineMap):
    """one field component in three dimensions"""
    def __init__(self, xRange, yRange, zRange, By3D, input_order='xyz', maxord=None, **kwargs):
        """"""
        assert len(xRange) == By3D.shape[0]
        assert len(yRange) == By3D.shape[1]
        assert len(zRange) == By3D.shape[2]
        assert yRange[0] == 0.0  # for symmetry assumptions (potential generation)

        ## reorder numbers
        required_order = 'yzx'
        change_order = [required_order.index(inp) for inp in input_order]
        yBy = moveaxis(By3D, range(3), change_order)
        shap = yBy.shape
        yBy = yBy.reshape((shap[0],-1))
        ## make polynomial in y direction
        yPoly = polyfit(yRange**2, yBy, deg=len(yRange)-1)
        yPoly = yPoly.reshape(shap)[::-1] # polynomial order starting with lowest first

        if maxord is None:
            maxord=2*(len(yRange)-1)+1 # was 4 before
            print(f'maxord auto-set to {maxord}')
        SplineMap.__init__(self, zRange, xRange, copy(yPoly[0]), maxord=maxord, **kwargs)
        self.yPoly = yPoly[1:]

        self.yLim = array([-yRange[-1], yRange[-1]])
        ## old
        #self.fldPrime = (fields[1]-fields[0]) / yRange[1]**2

    def scale_fields(self, factor):
        self.fld *= factor
        self.yPoly *= factor

    def potential(self, x, y, z):
        """return scaled magnetic potential so that nabla mag_pot = vec b."""
        zTrans, xTrans = self.transform.map(z, x)
        pot = y*rectBS(zTrans, xTrans, self.zPlane, self.xPlane, self.fld, 0, 0, self.maxord, self.smoothingFactor)
        for n, yp in enumerate(self.yPoly):
            pw = 3+2*n
            pot += y**pw / pw * rectBS(zTrans, xTrans, self.zPlane, self.xPlane, yp, 0, 0, self.maxord, self.smoothingFactor)
        return pot

    def potentialOnPath(self, x, y, z):
        inside = logical_and(in_ival(z, self.zLim), in_ival(x, self.xLim))
        inside = logical_and(in_ival(y, self.yLim), inside)
        return ag.select((inside,), (self.potential(x, y, z),))

def gridFromList(zList,xList,bList):
    Z, iz = unique(zList, return_inverse=True)
    X, ix = unique(xList, return_inverse=True)
    B=empty((len(Z), len(X)))
    B.fill(NaN)
    B[iz,ix] = bList
    return Z, X, B
