from scipy.special import legendre
import numpy as np
from .basis_function_2d import BasisFunction2D

class Basis2DLegendre(BasisFunction2D):
    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)
        self.degree = int(np.sqrt(num_shape_functions)) - 1
        # Precompute Legendre polynomials and their derivatives
        self.P = [legendre(n) for n in range(self.degree + 1)]
        self.dP = [P.deriv() for P in self.P]

    def value(self, xi, eta):
        xi = np.asarray(xi)
        eta = np.asarray(eta)
        num_points = xi.size
        values = np.zeros((self.num_shape_functions, num_points))
        index = 0
        for m in range(self.degree + 1):
            Pm_xi = self.P[m](xi)
            for n in range(self.degree + 1):
                Pn_eta = self.P[n](eta)
                values[index, :] = Pm_xi * Pn_eta
                index += 1
        return values

    def gradx(self, xi, eta):
        xi = np.asarray(xi)
        eta = np.asarray(eta)
        num_points = xi.size
        values = np.zeros((self.num_shape_functions, num_points))
        index = 0
        for m in range(self.degree + 1):
            dPm_xi = self.dP[m](xi)
            for n in range(self.degree + 1):
                Pn_eta = self.P[n](eta)
                values[index, :] = dPm_xi * Pn_eta
                index += 1
        return values

    def grady(self, xi, eta):
        xi = np.asarray(xi)
        eta = np.asarray(eta)
        num_points = xi.size
        values = np.zeros((self.num_shape_functions, num_points))
        index = 0
        for m in range(self.degree + 1):
            Pm_xi = self.P[m](xi)
            for n in range(self.degree + 1):
                dPn_eta = self.dP[n](eta)
                values[index, :] = Pm_xi * dPn_eta
                index += 1
        return values