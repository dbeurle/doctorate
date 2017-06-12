
import numpy as np

class LagrangeQ4:
    """ Simple implementation of Lagrange quadrilateral element """

    ξ_natural = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

    def __init__(self, isReducedQuadrature):

        self.sf = []
        self.femvals = []

        self.fillQuadraturePoints(isReducedQuadrature)

        for ξ, η in self.coords:
            self.evaluateShapeFunctions(ξ, η)

    def weights(self):
        return self.w

    def points(self):
        return len(self.w)

    def integrate(self, init, fn):

        for (i, (wl, femval)) in enumerate(zip(self.w, self.femvals)):
            init += fn(i, femval) * wl
        return init

    def fillQuadraturePoints(self, isReducedQuadrature):

        self.coords = []
        self.w = []

        if (isReducedQuadrature):
            self.coords.append([0.0, 0.0])
            self.w.append(4.0)
            return

        p = 1.0 / np.sqrt(3.0)
        self.coords.append([-p, -p])
        self.coords.append([ p, -p])
        self.coords.append([ p,  p])
        self.coords.append([-p,  p])
        self.w = [1.0, 1.0, 1.0, 1.0]

    def evaluateShapeFunctions(self, ξl, ηl):

        self.femvals.append([])
        self.sf.append([])

        for ξa, ηa in self.ξ_natural:

            # Shape functions at quadrature points
            self.sf[-1].append([0.25 * (1.0 + ξa * ξl) * (1.0 + ηa * ηl)])

            # Shape function derivatives at quadrature points
            self.femvals[-1].append([0.25 * ξa * (1.0 + ηl * ηa) , 0.25 * (1.0 + ξl * ξa) * ηa ])

    def shapeFunctions(self):
        return self.sf

    def femValues(self):
        return self.femvals

class LagrangeH8:

    """ Simple implementation of Lagrange quadrilateral element """

    ξ_natural = [[-1, -1, -1],
                 [ 1, -1, -1],
                 [ 1,  1, -1],
                 [-1,  1, -1],
                 [-1, -1,  1],
                 [ 1, -1,  1],
                 [ 1,  1,  1],
                 [-1,  1,  1]]

    def __init__(self, isReducedQuadrature):

        self.sf = []
        self.femvals = []

        self.fillQuadraturePoints(isReducedQuadrature)

        for ξ, η, ζ in self.coords:
            self.evaluateShapeFunctions(ξ, η, ζ)

    def weights(self):
        return self.w

    def points(self):
        return len(self.w)

    def integrate(self, init, fn):

        for (i, (wl, femval)) in enumerate(zip(self.w, self.femvals)):
            init += fn(i, femval) * wl
        return init

    def fillQuadraturePoints(self, isReducedQuadrature):

        self.coords = []
        self.w = []

        # Use the 1 quadrature scheme
        if (isReducedQuadrature):
            self.coords.append([0.0, 0.0, 0.0])
            self.w.append(8.0)
            return

        # Use the 2x2x2 quadrature scheme
        α = 1.0 / np.sqrt(3.0)
        self.coords = [[-α, -α, -α],
                       [ α, -α, -α],
                       [ α,  α, -α],
                       [-α,  α, -α],
                       [-α, -α,  α],
                       [ α, -α,  α],
                       [ α,  α,  α],
                       [-α,  α,  α]]
        self.w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        if len(self.coords) != len(self.w):
            exit()

    def evaluateShapeFunctions(self, ξl, ηl, ζl):

        self.femvals.append([])
        self.sf.append([])

        for ξa, ηa, ζa in self.ξ_natural:

            # Shape functions at quadrature points
            self.sf[-1].append([ 1.0/8.0 * (1.0 + ξa * ξl) * (1.0 + ηa * ηl) * (1.0 + ζa * ζl)])

            # Shape function derivatives at quadrature points
            self.femvals[-1].append( [1.0/8.0 * ξa * (1.0 + ηl * ηa) * (1.0 + ζl * ζa),
                                      1.0/8.0 * (1.0 + ξl * ξa) * ηa * (1.0 + ζl * ζa),
                                      1.0/8.0 * (1.0 + ξl * ξa) * (1.0 + ηl * ηa) * ζa] )

    def shapeFunctions(self):
        return self.sf

    def femValues(self):
        return self.femvals
