import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.integrate import trapz
import matplotlib.pyplot as plt

class Schrodinger(object):

    def __init__(self, **kwargs):
        self.set_constants(kwargs)
        self.set_coordinate(kwargs)
        self.psi0 = self.wavepacket(self.x, self.xa, self.k0x, self.a)
        self.V = self.potential(self.x, kwargs['potential'])
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

    def play_video(self, PSI):
        plt.ion()
        fig = plt.figure(figsize=(12, 9), facecolor='white')
        ax1 = fig.add_subplot(111, autoscale_on=False,
                            xlim=(self.x0, self.xf))
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$|\Psi|^2$')
        line, = ax1.plot(self.x, abs(PSI[:, 0]) ** 2, c='C0', lw=2)

        ax2 = ax1.twinx()
        ax2.plot(self.x, self.V, c='C1')
        ax2.set_xlim(self.x0, self.xf)
        ax2.set_ylim(0, 1.5 * self.V0)
        ax2.set_ylabel('$V$')
        ax2.fill_between(self.x,self.V,facecolor = 'C1',alpha =0.2)
        ax1.grid('on')
        plt.draw()
        for n in range(0, self.T - 1):
            line.set_ydata(abs(PSI[:, n]) ** 2)
            plt.pause(0.0001)
            plt.draw()
        plt.ioff()
        plt.show()

    def solve(self):
        U1, U2 = self.sparse_matrix()
        LU = scipy.sparse.linalg.splu(U1)
        PSI = np.zeros((self.J, self.T), dtype=complex)
        PSI[:, 0] = self.psi0

        for n in range(0, self.T - 1):
            b = U2.dot(PSI[:, n])
            psi = LU.solve(b)
            PSI[:, n + 1] = psi
            self.normalization[n] = trapz(abs(psi ** 2), self.x, dx=self.dx)
            self.averageX[n] = trapz(self.x * abs(psi ** 2), self.x, dx=self.dx)
            self.averageX2[n] = trapz(self.x ** 2 * abs(psi ** 2), self.x, dx=self.dx)
        return PSI

    def sparse_matrix(self):
        b = 1 + 1j * self.dt * self.hbar ** 2 / (2 * self.hbar \
                    * self.mass) * (1 / (self.dx * self.dx))  \
                    + 1j * self.dt * self.V / (2 * self.hbar)
        c = -1j * self.dt * self.hbar / (4 * self.mass * self.dx * self.dx) \
             * np.ones(self.J, dtype=complex)
        a = c
        d = 1 - 1j * self.dt * self.hbar ** 2 / (2 * self.hbar \
                    * self.mass) * (1 / (self.dx * self.dx)) \
                    - 1j * self.dt * self.V / (2 * self.hbar)
        U1 = np.array([a, b, c])
        U2 = np.array([-a, d, -c])
        diags = np.array([-1, 0, 1])
        return (scipy.sparse.spdiags(U1, diags, self.J, self.J).tocsc()\
            , scipy.sparse.spdiags(U2, diags, self.J, self.J).tocsc())

    def wavepacket(self, x, xa, k0x, a):
        f   = (1. / (np.pi * a ** 2)) ** 0.25
        e1  = np.exp(-((x - xa) ** 2.) / (2. * a ** 2))
        e2  = np.exp(1j * k0x * x)
        return f * e1 * e2

    def potential(self, x, U):
        def no_potential(x):
            return np.zeros(len(x))

        def delta_potential(x):
            V = np.zeros(len(x))
            V[int(len(x) / 2)] = 1
            return V

        def square_potential(x):
            return x ** 2 / 100

        def smooth_gaussian(x):
            V = 50*np.exp(-(x+7)**2.0/self.a**2.0) + 50*np.exp(-(x-(-7+14))**2.0/self.a**2.0)
            return V

        if U == 'delta':
            return delta_potential(x)
        elif U == 'square':
            return square_potential(x)
        elif U == 'smooth_gaussian':
            return smooth_gaussian(x)
        else:
            return no_potential(x)

    def set_constants(self, kwargs):
        self.mass = kwargs['mass']
        self.hbar = kwargs['hbar']

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
        self.V0 = kwargs['V0']
        self.Nt = int(round(self.tf / float(self.dt)))
        self.t = np.linspace(self.t0, self.Nt * self.dt, self.Nt)
        self.x = np.linspace(self.x0, self.xf, self.nx + 1)
        self.dx = self.x[1] - self.x[0]
        self.J = len(self.x)
        self.T = len(self.t)

args = {'nx': 1000, 
        'x0': -20, 
        'xf': 20, 
        'xa': -3, 
        't0': 0, 
        'tf': 6.0,
        'dt': 0.005, 
        'a': 1.5, 
        'k0x': 5,
        'V0': 50,
        'mass': 0.5,
        'hbar': 1,
        'potential': 'smooth_gaussian'}
        
schrodinger = Schrodinger(**args)
psi = schrodinger.solve()
schrodinger.play_video(psi)
schrodinger.plot_graphs()

