from autograd import elementwise_grad
from autograd.numpy import fabs
from numpy import abs, min, max, zeros, arange, linspace, zeros_like, pi
from numpy import array, eye, sin, cos
from numpy.linalg import cond, solve
from scipy.constants import c, value
from scipy.integrate import cumtrapz
from scipy.special import sindg, cosdg


def rigidityFromMeV(MeV, kinetic=False):
    """
    compute magnetic rigidity p/e (in T m) from particle momentum / energy.
    For ultrarelativistic cases as is the case here, E = cp.
    """
    if kinetic:
        total_MeV = value('electron mass energy equivalent in MeV') + MeV
        print('kinetic energy mode for rigidity')
    else:
        total_MeV = MeV
    return total_MeV * (1e6 / c)


class LinearTransform():
    def __init__(self, mirrorZ: bool=True):
        self.invRotMatrix = array([[1, 0], [0, 1]], float)
        self.invTranslation = array([0, 0], float)
        self.mirrorZ = mirrorZ

    def translate(self, deltaZ, deltaX):
        # performs inverse translation
        self.invTranslation -= [deltaZ, deltaX]

    def rotate(self, angleCCW, aroundGlobalCenter=False, deg=True):
        # performs inverse rotation, sa the inverse map is required for tracking
        if deg:
            si, co = sindg(angleCCW), cosdg(angleCCW)
        else:
            si, co = sin(angleCCW), cos(angleCCW)
        iro = array([[co, si], [-si, co]])
        self.invRotMatrix = iro @ self.invRotMatrix
        if aroundGlobalCenter:
            ro = array([[co, -si], [si, co]])
            self.invTranslation = ro @ self.invTranslation

    def shear(self, lamb, inZ=True):
        ish = eye(2,dtype=float)
        if inZ:
            print('shear in Z direction, assuming you know what you are doing')
            ish[0,1] = lamb
        else: #
            print('shear in X direction, assuming you know what you are doing')
            ish[1,0] = lamb
        self.invRotMatrix = ish @ self.invRotMatrix


    def map(self, Z, X):
        sZ = Z + self.invTranslation[0]
        sX = X + self.invTranslation[1]
        Zed = self.invRotMatrix[0, 0] * sZ + self.invRotMatrix[0, 1] * sX
        Ex = self.invRotMatrix[1, 0] * sZ + self.invRotMatrix[1, 1] * sX
        if self.mirrorZ:
            return fabs(Zed), Ex
        else:
            return Zed, Ex
               


class AbstractField():
    def __init__(self, normalized: bool):
        self.transform = LinearTransform()
        self.normalizedToRigidity = normalized
        
    def _normalize(self, rigidity):
        # to be overwritten by the respective sub-class
        pass

    def normalize(self, rigidity):
        if not self.normalizedToRigidity:
            self._normalize(rigidity)
            self.normalizedToRigidity = True
        else:
            raise Exception('Field is already normalized - you can only normalize once.')



