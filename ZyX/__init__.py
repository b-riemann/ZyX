"""
ZX plane (machine plane) motion of (reference) particles,
computation of focusing terms using Automatic Differentiation.

Bernard Riemann (2019/2020)
"""
from numpy import absolute, allclose, amin, amax, asarray, zeros, arange, linspace, \
                  zeros_like, pi, ones_like, mean, diff, floor, empty, array, concatenate, \
                  sin, cos, meshgrid, trapz, copy, logical_and, sum, argmin, argmax, Inf
from scipy.integrate import cumtrapz
from scipy.special import factorial
from scipy.optimize import root_scalar, minimize_scalar
from pandas import DataFrame, Series

from .basic import AbstractField, rigidityFromMeV

class Trajectory():
    def __init__(self, Z, X, s, theta, curv=None, startSymmetric=True):
        self.s = copy(asarray(s).reshape([-1]))
        self.Z = copy(asarray(Z).reshape([-1]))
        self.X = copy(asarray(X).reshape([-1]))
        self.theta = copy(asarray(theta).reshape([-1]))
        self.genCTheta()
        self.curv = copy(asarray(curv).reshape([-1]))

        self.bFocus = None
        self.kFocus = None
        self.startSymmetric = startSymmetric

    def genCTheta(self):
        """assumes initial angle is zero (symmetric)"""
        self.cTheta = zeros_like(self.s)
        if len(self.s) > 1:
            self.cTheta[1:] = (self.theta[:-1] + self.theta[1:]) / 2
            self.cThetaSin = sin(self.cTheta)
            self.cThetaCos = cos(self.cTheta)

    def __len__(self):
        return self.s.size

    def track(self, sDiff: float, nSteps: int, curv_func, scale: float=1.0):
        nOld = len(self)
        nNew = nOld + nSteps
        self.s.resize(nNew)
        self.s[nOld:] = self.s[nOld-1] + arange(1, nSteps+1)*sDiff
        self.Z.resize(nNew)
        self.X.resize(nNew)
        self.curv.resize(nNew)
        self.theta.resize(nNew)
        for n in range(nOld, nNew):
            zNow = self.Z[n] = self.Z[n - 1] + sDiff * cos(self.theta[n - 1])
            xNow = self.X[n] = self.X[n - 1] - sDiff * sin(self.theta[n - 1])
            self.curv[n] = scale*curv_func(zNow, xNow)
            self.theta[n] = self.theta[n - 1] + sDiff * self.curv[n]
        self.genCTheta()

    def sqDistanceToPoint(self, pointZ, pointX, normalize=True, mm=True):
        "assumes ordered Z"
        posidx=argmax(self.Z>pointZ)
        #trajEdge = (self.Z[posidx-1]-self.Z[posidx], self.X[posidx-1]-self.X[posidx])
        #pointEdge = (pointZ-self.Z[posidx], pointX-self.X[posidx])
        # cross product
        edgeZ = (self.Z[posidx-1]-self.Z[posidx])*1000
        edgeX = (self.X[posidx-1]-self.X[posidx])*1000
        cross_prod = 1000*((pointZ-self.Z[posidx])*edgeX - (pointX-self.X[posidx])*edgeZ)
        cross_prod *= cross_prod
        if normalize:
            cross_prod /= edgeX**2 + edgeZ**2
        if not mm:
            cross_prod /= 1000
        return cross_prod

    def plot(self, ax, whiteBg=True, linewidth=0.9, mirrorZ=False, **kwargs):
        zed = -self.Z if mirrorZ else self.Z
        if whiteBg:
            ax.plot(zed, self.X, linewidth=2.0, color='white')
        ax.plot(zed, self.X, linewidth=linewidth, **kwargs)
        ax.set(xlabel='Z [m]', ylabel='X [m]')

    def angle(self, n=-1, deg=True):
        ang = self.cTheta[n]
        return ang * 180 / pi if deg else ang

    def indexAtS(self, thisS):
        return argmax(self.s>thisS)

    def __repr__(self):
        return 'orbit trajectory starting (Z=%.2e m, X=%.2e m) ending (Z=%.2e m, X=%.2e m) with %.3f deg' \
               % (self.Z[0], self.X[0], self.Z[-1], self.X[-1], self.angle(deg=True))

    def toFrame(self):
        df = DataFrame()
        for key in ('s', 'Z', 'X', 'cTheta'):
            df[key] = getattr(self, key)
        df['cnZ'] = cos(self.cTheta)
        df['cnX'] = -sin(self.cTheta)
        if self.bFocus is not None:
            df = self.bFocus.toFrame(df)
        return df

def parallelTrajectory(s=arange(0,1,0.001), xOffset=0.0):
    z = arange(0,0.7,0.001)
    x = ones_like(z)*xOffset
    return Trajectory(z, x, z, zeros_like(z))


class CurvatureFunction2D():
    def __init__(self, beamEnergyMeV):
        self.subFields = list()
        self.rigidity = rigidityFromMeV(beamEnergyMeV)
        self.trajectories = list()

    def addField(self, field: AbstractField):
        field.normalize(self.rigidity)
        self.subFields.append(field)

    def curvatureInPlane(self, z, x):
        #b = zeros((z.size, x.size), dtype=float)
        zMesh, xMesh = [arr.T.flatten() for arr in meshgrid(z, x)]
        b = zeros_like(zMesh)
        for subField in self.subFields:
            b += subField.onPath(zMesh, xMesh)
        return b.reshape((z.size, x.size))

    def curvatureOnPath(self, z, x):
        b = zeros(z.size, dtype=float)
        for subField in self.subFields:
            b += subField.onPath(z, x)
        return b

    def potentialOnPath(self, x, y, z):
        pot = zeros(z.size, dtype=float)
        for subField in self.subFields:
            try:
                pot += subField.potentialOnPath(x, y, z)
            except NameError:
                raise RuntimeError('at least one subfield is not 3d')
        return pot

    def plot(self, ax, z=None, x=None, vlim=None, cmap='BrBG',
             trajectories=None):  # z=arange(-0.25, 1, 0.005), x=arange(-0.02, 0.02, 0.002)):
        if z is None:
            lim = zeros((2, 2), dtype=float)  # array([[0,1e-3],[-5e-3,5e-3]], float)  #xlim, ylim
            try:
                for trajectory in trajectories:
                    trajectory.plot(ax)
                    traj = array(((amin(trajectory.Z), amax(trajectory.Z)), (amin(trajectory.X), amax(trajectory.X))))
                    for n in range(2):
                        if traj[n, 0] < lim[n, 0]:
                            lim[n, 0] = traj[n, 0]
                        if traj[n, 1] > lim[n, 1]:
                            lim[n, 1] = traj[n, 1]
                lim += [[0, 1e-3], [-5e-3, 5e-3]]
                z = linspace(lim[0, 0], lim[0, 1], 250)
                x = linspace(lim[1, 0], lim[1, 1], 20)
            except TypeError:
                raise Exception('generate a trajectory for map limits.')

        fld = self.curvatureInPlane(z, x).T
        if vlim is None:
            vlim = amax(absolute(fld))
        levels = vlim * arange(-20.5, 21) / 20  #arange(-10.5, 11) / 10
        quad = ax.contourf(z, x, fld, levels, cmap=cmap)  # vmin=-vlim, vmax=vlim
        # cBar = colorbar(quad)
        # cBar.set_label('curvature $\cdot$ m')

    def solveTrajectory(self, X0: float, sMax: float, sDiff: float=1e-3, scale: float=1.0):
        """
        generate a Trajectory starting at (X=X0,Z=0) with edge length sDiff and up to length sMax
        """
        curv0 = scale*self.curvatureOnPath(asarray(0.0), asarray(X0))
        traj = Trajectory(0.0, X0, 0.0, curv0*sDiff/2, curv0)
        s = arange(0, sMax + sDiff / 2, sDiff)
        traj.track(sDiff, len(s)-1, self.curvatureOnPath, scale)
        return traj

    def findXforAngle(self, designAngleDeg: float, sMax: float, sDiff: float=1e-3, xSearchInterval=(-0.01, 0.01), **kwargs):
        """
        generate a Trajectory ending with designAngleDeg with edge length dFiff and up to length sMax. (optimizes using .solveTrajectory)
        """
        def angleDiff(X: float) -> float:
            traj = self.solveTrajectory(X, sMax, sDiff)
            return traj.angle(deg=True)-designAngleDeg
        result = root_scalar(angleDiff, bracket=xSearchInterval, **kwargs)
        print(result)
        return self.solveTrajectory(result.root, sMax, sDiff)

    def findXthruPoint(self, pointZX: iter, sMax: float, sDiff: float=1e-3, **kwargs):
        """
        generate a Trajectory running through pointZX with edge length dFiff and up to length sMax. (optimizes using .solveTrajectory)
        """
        def squareDist(X: float) -> float:
            traj = self.solveTrajectory(X, sMax, sDiff)
            idex = argmin(absolute(traj.Z-pointZX[0]))
            return (traj.Z[idex]-pointZX[0])**2 + (traj.X[idex]-pointZX[1])**2
        result = minimize_scalar(squareDist, **kwargs)
        print(result)
        return self.solveTrajectory(result.x, sMax, sDiff)

def dummyKickDrift(filename, length = 0.2025, iRho = 1.49969e-01, N=32):
    sliceLength = length/N
    with open(filename, 'w') as f:
        f.write("# kick-drift export (upright multipoles) for tracy-null\norder 2\n")
        f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000\ndrif %.8e\n' % (sliceLength/2, iRho, sliceLength))
        for n in range(1,N):
            f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000\ndrif %.8e\n' % (sliceLength, iRho, sliceLength))
        f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000' % (sliceLength/2, iRho))
