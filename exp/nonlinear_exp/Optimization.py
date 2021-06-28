##################################
# Naive implementation of optimization routines with derivative observations.
# Algorithms use code from scipy.optimize with replaced step direction determined
# by a GP estimate
#
#   --- Code only intended as a proof-of-concept ---
#
##################################

import numpy as np
import warnings
from scipy.optimize import OptimizeResult
# import scipy.optimize.linesearch as linesearch
from scipy.optimize.linesearch import LineSearchWarning,line_search_wolfe1,line_search_wolfe2


# Import GP-derivative code
import sys
sys.path.append("../../src")
from inference import DerivativeGaussianProcess


# Standard status message from scipy.optimize
_status_message = {
    "success": "Optimization terminated successfully.",
    "maxfev": "Maximum number of function evaluations has " "been exceeded.",
    "maxiter": "Maximum number of iterations has been " "exceeded.",
    "pr_loss": "Desired error not necessarily achieved due " "to precision loss.",
    "nan": "NaN result encountered.",
    "out_of_bounds": "The result is outside of the provided " "bounds.",
}


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret



def GPHessianMinimizer(
    fun,
    x0,
    kern,
    jac,
    memory=2,
    nugget=0.0,
    args=(),
    callback=None,
    gtol=1e-5,
    norm=np.inf,
    maxiter=None,
    la=False,
    disp=False,
    return_all=False,
    **unknown_options
):
    #############
    # The inputs follow the notation 
    #############
    x0 = np.asarray(x0).reshape(-1, 1)
    if maxiter is None:
        maxiter = len(x0) * 200

    if return_all:
        allvecs = [x0]

    warnflag = 0

    xi = np.copy(x0)
    old_fval = fun(xi[:,0])
    old_g = jac(xi[:,0]).reshape(-1,1)

    dX = np.copy(xi)
    dY = np.copy(old_g.reshape(-1,1))

    DIM = x0.shape[0]

    GRAD_COUNT = 0
    FUN_COUNT = 0

    di = -old_g
    scale = np.linalg.norm(old_g) / 2

    old_old_fval = old_fval + scale

    DGP = DerivativeGaussianProcess(kern, nugget=nugget)

    gnorm = np.linalg.norm(old_g, ord=norm)

    i = 0
    while (gnorm > gtol) and (i < maxiter):

        try:
            # alpha, fc, gc, old_fval, old_old_fval, new_g = _line_search_wolfe12(f, myfprime, xk, pk, gfk,old_fval, old_old_fval, amin=1e-100, amax=1e100)
            alpha, fc, gc, old_fval, old_old_fval, new_g = _line_search_wolfe12(fun, jac, xi[:,0], di[:,0], old_g[:,0],old_fval, old_old_fval, amin=1e-100, amax=1e100)

        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break





        
        xi += alpha * di


        new_g = new_g.reshape(-1, 1)

        if callback is not None:
            callback(xi)

        if return_all:
            allvecs.append(np.copy(xi))

        i += 1

        gnorm = np.linalg.norm(new_g, ord=norm)
        if gnorm <= gtol:
            break


        DGP.condition(dX, dY)
        Hi = DGP.infer_h(xi)

        di = -Hi.invmul(new_g)

        if di.T @ new_g > 0:
            di *= -1

        dX = np.hstack((dX, xi))[:, -memory:]
        dY = np.hstack((dY, new_g))[:, -memory:]

        old_g = new_g

        GRAD_COUNT += gc
        FUN_COUNT += fc

    fval = old_fval

    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif i >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xi).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % i)
        print("         Function evaluations: %d" % FUN_COUNT)
        print("         Gradient evaluations: %d" % GRAD_COUNT)

    result = OptimizeResult(
        fun=fval,
        jac=new_g,
        nfev=FUN_COUNT,
        njev=GRAD_COUNT,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xi,
        nit=i,
    )
    if return_all:
        result["allvecs"] = allvecs

    return result


def GPGradientMinimizer(
    fun,
    x0,
    kern,
    jac,
    memory=2,
    nugget=0.0,
    args=(),
    callback=None,
    gtol=1e-5,
    norm=np.inf,
    fixed_w=False,
    la=False,
    maxiter=None,
    disp=False,
    return_all=False,
    **unknown_options
):

    x0 = np.asarray(x0).reshape(-1, 1)
    if maxiter is None:
        maxiter = len(x0) * 200

    if return_all:
        allvecs = [x0]

    warnflag = 0

    xi = np.copy(x0)
    old_fval = fun(xi)
    old_g = jac(xi)

    dX = np.copy(xi)
    dY = np.copy(old_g)

    DIM = x0.shape[0]

    min_grad = np.zeros((DIM, 1))

    GRAD_COUNT = 0
    FUN_COUNT = 0

    di = -old_g

    scale = np.linalg.norm(old_g) / 2

    old_old_fval = old_fval + scale

    DGP = DerivativeGaussianProcess(kern, nugget=nugget)

    gnorm = np.linalg.norm(old_g, ord=norm)

    i = 0
    while (gnorm > gtol) and (i < maxiter):

        try:
            alpha, fc, gc, old_fval, old_old_fval, new_g = _line_search_wolfe12(fun, jac, xi[:,0], di[:,0], old_g[:,0],old_fval, old_old_fval, amin=1e-100, amax=1e100)

        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break



        xi += alpha * di


        new_g = new_g.reshape(-1, 1)

        if callback is not None:
            callback(xi)

        if return_all:
            allvecs.append(np.copy(xi))

        i += 1


        gnorm = np.linalg.norm(new_g, ord=norm)
        if gnorm <= gtol:
            break

        if not fixed_w:
            upd = 2 / np.linalg.norm(old_g) ** 2
            kern.update_hyperparameter("w", upd * np.ones((DIM, 1)))

        dX = np.hstack((dX, xi))[:, -memory:]
        dY = np.hstack((dY, new_g))[:, -memory:]

        DGP.condition(dY, dX-xi)


        di = DGP.infer_g(min_grad)


        if di.T @ new_g > 0:
            di *= -1

        old_g = new_g

        GRAD_COUNT += gc
        FUN_COUNT += fc

    fval = old_fval

    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif i >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xi).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % i)
        print("         Function evaluations: %d" % FUN_COUNT)
        print("         Gradient evaluations: %d" % GRAD_COUNT)

    result = OptimizeResult(
        fun=fval,
        jac=new_g,
        nfev=FUN_COUNT,
        njev=GRAD_COUNT,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xi,
        nit=i,
    )
    if return_all:
        result["allvecs"] = allvecs

    return result
