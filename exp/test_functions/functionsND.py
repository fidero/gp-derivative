import numpy as np
from .function import functionND
import matplotlib.pyplot as plt


# returning function by number (for simplicity, instead of tyoing in the name)


def byNumber(number,**kwargs):
    if number == 1:
        return Quadratic(**kwargs)
    elif number == 2:
        return Cigar(**kwargs)
    elif number == 3:
        return Discus(**kwargs)
    elif number == 4:
        return Ellipsoid(**kwargs)
    elif number == 5:
        return DiffPowers(**kwargs)
    elif number == 6:
        return Rosenbrock(**kwargs)
    elif number == 7:
        return SC2(**kwargs)
    elif number == 8:
        return Schwefel21(**kwargs)
    elif number == 9:
        return Schwefel12(**kwargs)
    elif number == 10:
        return L1(**kwargs)

    else:
        raise ValueError(f"No function defined for #{number}")
    
        
    



#---------Quadratic----------------------------------------------------------------
class Quadratic(functionND):

    def __init__(self,Q,xsol=None,x0=None,**kwargs):

        Q = np.atleast_1d(Q)
        if Q.ndim == 1:
            self.Q = np.diag(np.array(Q))
        elif Q.ndim == 2:
            self.Q = Q

        N=Q.shape[0]

        super().__init__(N=N,**kwargs)
        if x0 is None:
            self.x0 = -3*np.ones(self.dim,dtype=np.float)
        else:
            self.x0 = np.asarray(x0)
        if xsol is None:
            self.global_min = np.sqrt(2)*np.ones(self.dim,dtype=np.float)
        else:
            self.global_min = xsol
    
    def _f(self,x):
        d = x-self.global_min
        return 0.5*d.dot((d.dot(self.Q)).T)

    def _g(self,x):
        d = x-self.global_min

        return d.dot(self.Q)



#---------Cigar----------------------------------------------------------------
class Cigar(Quadratic):

    def __init__(self,N,alpha=10.0,**kwargs):
        self.alpha=alpha
        Q=np.eye(N)
        Q[0,0]*=alpha

        super().__init__(Q=Q,**kwargs)


#---------Discus----------------------------------------------------------------
class Discus(Quadratic):

    def __init__(self,N,alpha=10.0,**kwargs):
        self.alpha=alpha
        Q=alpha*np.eye(N)
        Q[0,0]=1.0
        super().__init__(Q=Q,**kwargs)


#---------Ellipsoid----------------------------------------------------------------
class Ellipsoid(Quadratic):

    def __init__(self,N,alpha=10.0,**kwargs):
        self.alpha=alpha
        d = np.power(alpha,(np.arange(N)+1)/N)
        Q=np.diag(d)
        super().__init__(Q=Q,**kwargs)


#---------DiffPowers----------------------------------------------------------------------------------------------------
class DiffPowers(functionND):
    def __init__(self, N,**kwargs):
        super().__init__(N=N,**kwargs)

        # self.x0 = np.array([-0.5, 1.5],dtype=np.float)
        # self.x0 = np.power(-np.ones(N),np.arange(N)+1)
        self.x0 = np.ones(N)
        self.x0[-1]*=-1

        # minima=np.zeros(N)
        minima = np.ones(N)
        self.global_min = [minima]
        self.exp=2+10*np.arange(self.dim)/self.dim

    def _f(self, x):
        x_=self.global_min[0]
        # s = np.sign(x-x_)
        d = np.abs(x-x_)
        
        dpow= np.power(d,self.exp)
        return np.sum(dpow,axis=0)

    def _g(self, x):
        x_=self.global_min[0]
        s = np.sign(x-x_)
        d = np.abs(x-x_)
        
        der= s*self.exp*np.power(d,self.exp-1)
        return der




#---------Rosenbrock----------------------------------------------------------------------------------------------------


class Rosenbrock(functionND):
    def __init__(self, N,a=1.0,b=100.0,**kwargs):
        super().__init__(N=N,**kwargs)

        self.x0 = np.ones(N)
        self.x0[-1]*=-1

        self.a = a
        self.b = b
        minima=self.a**2*np.ones(N,dtype=np.float)
        minima[0]=self.a
        self.global_min = [minima]


    def _f(self, x):
        a = self.a
        b = self.b
        x=np.asarray(x)
        r = np.sum(b * (x[1:] - x[:-1]**2.0)**2.0 + (a - x[:-1])**2.0,
                  axis=0)
        return r

    def _g(self, x):
        a = self.a
        b = self.b

        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (2*b * (xm - xm_m1**2) -
                     4*b * (xm_p1 - xm**2) * xm - 2 * (a - xm))
        der[0] = -4*b * x[0] * (x[1] - x[0]**2) - 2 * (a - x[0])
        der[-1] = 2*b * (x[-1] - x[-2]**2)
        return der


#---------StriclyConvex2---------------------------------------------------------------------------------------------------


class SC2(functionND):
    def __init__(self, N,a=10.0,**kwargs):
        super().__init__(N=N,**kwargs)

        self.x0 = np.ones(N,dtype=np.float)

        self.a = a
        minima=np.zeros(N,dtype=np.float)
        self.global_min = [minima]

    def _f(self, x):
        a = self.a
        x=np.asarray(x)
        # r = np.sum(np.arange(1,self.dim+1)*(np.exp(x)-x),axis=0)/a
        # r = np.arange(1,self.dim+1).dot(np.exp(x)-x)/a
        r = np.einsum('i,i...',np.arange(1,self.dim+1),np.exp(x)-x)/a #np.arange(1,self.dim+1).dot(np.exp(x)-x)/a
        return r

    def _g(self, x):
        a = self.a

        x = np.asarray(x)
        der = np.arange(1,self.dim+1)*(np.exp(x) - 1.0)/a

        return der



#---------Schwefel21----------------------------------------------------------------------------------------------------

class Schwefel21(functionND):
    def __init__(self, N, a=1, **kwargs):
        super().__init__(N=N, **kwargs)

        self.x0 = 2*np.ones(N)
        self.x0[-1]*=-1
        self.a=a*np.ones(N)
        self._min=np.zeros(N,dtype=np.float)

        self.global_min = [self._min]


    def _f(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)


        # x_=a*(x-m)
        # x_=np.abs(x-m)
        x_=np.abs(a*(x-m))
        return np.sum(x_,axis=0) + np.prod(x_,axis=0)

    def _g(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)
        # x_=(x-self._min)*self.a
        x_=(x-m)*a
        xabs=np.abs(x_)
        c = np.prod(xabs,axis=0)
        der = a*(x_/xabs)*(1  + c/xabs)
        
        return der

#---------Schwefel12----------------------------------------------------------------------------------------------------

class Schwefel12(functionND):
    def __init__(self, N,a=1,**kwargs):
        super().__init__(N=N,**kwargs)

        self.x0 = 1*np.ones(N)
        self.x0[0]*=-2.5
        self.a=a*np.ones(N)
        self._min=np.zeros(N,dtype=np.float)

        self.global_min = [self._min]


    def _f(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)

        x_=a*(x-m)
        d=np.cumsum(x_,axis=0)
        return np.sum(d**2,axis=0)/2

    def _g(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)

        x_=a*(x-m)
        d_rev= np.flip(np.cumsum(x_,axis=0),axis=0)
        der =self.a*np.flip(np.cumsum(d_rev,axis=0),axis=0)
        return der

    def _hp(self, x,p):
        x=np.asarray(x)
        p=np.asarray(p)
        m=self._min
        a=self.a
        if p.ndim>1:
            sh=[1]*p.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)

        p_=a*p
        d_rev= np.flip(np.cumsum(p_,axis=0),axis=0)
        der =a*np.flip(np.cumsum(d_rev,axis=0),axis=0)
        return der


#---------L1----------------------------------------------------------------------------------------------------

class L1(functionND):
    def __init__(self, N,a=1,**kwargs):
        super().__init__(N=N,**kwargs)

        self.x0 = 1*np.ones(N)
        self.x0[0]*=-2.5
        self.a=a*np.ones(N)
        self._min=np.zeros(N,dtype=np.float)

        self.global_min = [self._min]


    def _f(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)


        x_=np.abs(a*(x-m))
        return np.sum(x_,axis=0)

    def _g(self, x):
        x=np.asarray(x)
        m=self._min
        a=self.a
        if x.ndim>1:
            sh=[1]*x.ndim
            sh[0]=-1
            a=a.reshape(sh)
            m=m.reshape(sh)
        # x_=(x-self._min)*self.a
        x_=(x-m)*a
        xabs=np.abs(x_)
        der = a*(x_/xabs)
        return der



if __name__ == '__main__':

    N=50
    v = np.arange(N).reshape(-1,1)

    Q=np.linalg.inv(np.power(0.9,np.abs(v-v.T)))

    for i in range(2,11):
        fun = byNumber(i,N=N)
        fun.test_grad(eps=1e-3)

    # i=10
    # fun = byNumber(i,N=N,a=np.random.randn(N)+1)
    # fun.test_grad(eps=1e-3)

    # for i in range(1,10):

    #     fun = byNumber(i,noise_f_std=0.0, noise_grad_std=0.0)
    #     print(fun)
    #     fun.test_grad(eps=1e-5)


    # print(fun)
    # fmin=np.min(fun.global_fmin)
    # print(fmin)
    # print(fun._g(fun.x0))


    # limits = fun.plotlimits
    # x = np.linspace(*limits[0], 100)
    # y = np.linspace(*limits[1], 90)

    # X, Y = np.meshgrid(x, y)

    # Z = fun.grid_f(X, Y)
    # Z = np.log(Z-fmin + 1)

    

    # plt.contour(X, Y, Z)
    # plt.show()
