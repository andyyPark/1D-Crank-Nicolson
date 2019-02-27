import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.integrate import trapz
import matplotlib.pyplot as plt

class Schrodinger(object):

    def __init__(self, **kwargs):
        self.set_constants()
        self.set_coordinate(kwargs)
        self.psi0 = self.wavepacket(self.x, self.xa, self.k0x, self.a)
        self.V = self.potential(self.x)
        self.normalization = np.zeros(self.T)
        self.averageX = np.zeros(self.T)
        self.averageX2 = np.zeros(self.T)

    def plot_graphs(self):
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.normalization)
        plt.title('Normalization')
        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.averageX)
        plt.title('Average x')
        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.averageX2)
        plt.title('Average x^2')
        plt.show()

    def solve(self, video=False):
        U1, U2 = self.sparse_matrix()
        LU = scipy.sparse.linalg.splu(U1)
        PSI = np.zeros((self.J, self.T), dtype=complex)
        PSI[:, 0] = self.psi0

        if video:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line, = ax.plot(self.x, abs(PSI[:, 0]) ** 2)
            ax.plot(self.x, self.V)
            ax.grid('on')
            plt.draw()

        for n in range(0, self.T - 1):
            b = U2.dot(PSI[:, n])
            psi = LU.solve(b)
            PSI[:, n + 1] = psi
            self.normalization[n] = trapz(abs(psi ** 2), self.x, dx=self.dx)
            self.averageX[n] = trapz(self.x * abs(psi ** 2), self.x, dx=self.dx)
            self.averageX2[n] = trapz(self.x ** 2 * abs(psi ** 2), self.x, dx=self.dx)
            if video:
                line.set_ydata(abs(psi) ** 2)
                plt.pause(0.0001)
                plt.draw()
        if video:
            plt.ioff()
            plt.show()

    def sparse_matrix(self):
        b = 1 + 1j * self.dt * self.hbar ** 2 * (1 / (self.dx * self.dx)) \
        + 1j * self.dt * self.V / (2 * self.hbar)
        c = -1j * self.dt * self.hbar / (4 * self.mass * self.dx * self.dx) \
             * np.ones(self.J, dtype=complex)
        a = c
        d = 1 - 1j * self.dt * self.hbar ** 2 * (1 / (self.dx * self.dx)) \
        + 1j * self.dt * self.V / (2 * self.hbar)
        U1 = [a, b, c]
        U2 = [-a, d, -c]
        diags = [-1, 0, 1]
        return (scipy.sparse.spdiags(U1, diags, self.J, self.J).tocsc()\
            , scipy.sparse.spdiags(U2, diags, self.J, self.J).tocsc())

    def wavepacket(self, x, xa, k0x, a):
        f   = (1. / (2 * np.pi * a ** 2)) ** 0.25
        e1  = np.exp(-((x - xa) ** 2.) / (4. * a ** 2))
        e2  = np.exp(1j * k0x * x)
        return f * e1 * e2

    def potential(self, x):
        return np.zeros(len(x))

    def set_constants(self, mass=0.5, hbar=1):
        self.mass = mass
        self.hbar = hbar

    def set_coordinate(self, kwargs):
        self.nx = kwargs['nx']
        self.x0 = kwargs['x0']
        self.xf = kwargs['xf']
        self.xa = kwargs['xa']
        self.t0 = kwargs['t0']
        self.tf = kwargs['tf']
        self.dt = kwargs['dt']
        self.a = kwargs['a']
        self.k0x = kwargs['k0x']
        self.Nt = int(round(self.tf / float(self.dt)))
        self.t = np.linspace(self.t0, self.Nt * self.dt, self.Nt)
        self.x = np.linspace(self.x0, self.xf, self.nx)
        self.dx = self.x[1] - self.x[0]
        self.J = len(self.x)
        self.T = len(self.t)

args = {'nx': 1000, 'x0': -50, 'xf': 50, 'xa': 3, 't0': 0, 'tf': 20,
        'dt': 0.005, 'a': 1.2, 'k0x': 3}
schrodinger = Schrodinger(**args)
schrodinger.solve(video=True)
schrodinger.plot_graphs()

