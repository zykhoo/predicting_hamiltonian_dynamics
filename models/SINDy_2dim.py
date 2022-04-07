# using SINDy to find the derivative of the system
# this code is adapted from pysindy by dynamicslab (https://github.com/dynamicslab/pysindy)
# instead of using trajectories of x, we use x and xdot to find an equation that describes the dynamics of the system

"""
Base class for SINDy optimizers.
"""
import abc

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_X_y

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import check_is_fitted

from itertools import repeat

from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_array

from sklearn.base import TransformerMixin

from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from sklearn.utils import check_array

import warnings
from typing import Sequence

from numpy import isscalar
from numpy import ndim
from numpy import newaxis
from numpy import vstack
from numpy import zeros
from scipy.integrate import odeint
from scipy.linalg import LinAlgWarning
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class ComplexityMixin:
    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)


class BaseOptimizer(LinearRegression, ComplexityMixin):
    """
    Base class for SINDy optimizers. Subclasses must implement
    a _reduce method for carrying out the bulk of the work of
    fitting a model.
    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.
    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.
    """

    def __init__(self, max_iter=20, normalize=False, fit_intercept=False, copy_X=True):
        super(BaseOptimizer, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X
        )

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.max_iter = max_iter
        self.iters = 0
        self.coef_ = None
        self.ind_ = None
        self.history_ = []

    # Force subclasses to implement this
    @abc.abstractmethod
    def _reduce(self):
        """
        Carry out the bulk of the work of the fit function.
        Subclass implementations MUST update self.coef_.
        """
        raise NotImplementedError

    def fit(self, x_, y, sample_weight=None, **reduce_kws):
        """
        Fit the model.
        Parameters
        ----------
        x_ : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        sample_weight : float or numpy array of shape (n_samples,), optional
            Individual weights for each sample
        reduce_kws : dict
            Optional keyword arguments to pass to the _reduce method
            (implemented by subclasses)
        Returns
        -------
        self : returns an instance of self
        """
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.iters = 0
        self.ind_ = np.ones((y.shape[1], x.shape[1]), dtype=bool)
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0].T  # initial guess
        self.history_.append(self.coef_)

        self._reduce(x, y, **reduce_kws)
        self.ind_ = np.abs(self.coef_) > 1e-14

        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class _MultiTargetLinearRegressor(MultiOutputRegressor, ComplexityMixin):
    @property
    def coef_(self):
        return np.vstack([est.coef_ for est in self.estimators_])

    @property
    def intercept_(self):
        return np.array([est.intercept_ for est in self.estimators_])

class STLSQ(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.
    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + alpha \\times \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight that are below a given threshold.
    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.
    alpha : float, optional (default 0.1)
        Optional L2 (ridge) regularization on the weight vector.
    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.
    ridge_kw : dict, optional
        Optional keyword arguments to pass to the ridge regression.
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.
    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.
    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s)
    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.1,
        max_iter=20,
        ridge_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
    ):
        super(STLSQ, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.alpha = alpha
        self.ridge_kw = ridge_kw

    def _sparse_coefficients(self, dim, ind, coef, threshold):
        """Perform thresholding of the weight vector(s)
        """
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x, y):
        """Perform the ridge regression
        """
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding
        """
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Iterates the thresholding. Assumes an initial guess is saved in
        self.coef_ and self.ind_
        """
        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(x[:, ind[i]], y[:, i])
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )
                coef[i] = coef_i
                ind[i] = ind_i

            self.history_.append(coef)
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )

# Define a special object for the default value of t in
# validate_input. Normally we would set the default
# value of t to be None, but it is possible for the user
# to pass in None, in which case validate_input performs
# no checks on t.
T_DEFAULT = object()


def validate_input(x, t=T_DEFAULT):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    check_array(x)

    if t is not T_DEFAULT:
        if t is None:
            raise ValueError("t must be a scalar or array-like.")
        # Apply this check if t is a scalar
        elif np.ndim(t) == 0:
            if t <= 0:
                raise ValueError("t must be positive")
        # Only apply these tests if t is array-like
        elif isinstance(t, np.ndarray):
            if not len(t) == x.shape[0]:
                raise ValueError("Length of t should match x.shape[0].")
            if not np.all(t[:-1] < t[1:]):
                raise ValueError("Values in t should be in strictly increasing order.")
        else:
            raise ValueError("t must be a scalar or array-like.")

    return x


def drop_nan_rows(x, x_dot):
    x = x[~np.isnan(x_dot).any(axis=1)]
    x_dot = x_dot[~np.isnan(x_dot).any(axis=1)]
    return x, x_dot


def prox_l0(x, threshold):
    """Proximal operator for l0 regularization."""
    return x * (np.abs(x) > threshold)


def prox_l1(x, threshold):
    """Proximal operator for l1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# TODO: replace code block with proper math block
def prox_cad(x, lower_threshold):
    """
    Proximal operator for CAD regularization
    .. code ::
        prox_cad(z, a, b) =
            0                    if |z| < a
            sign(z)(|z| - a)   if a < |z| <= b
            z                    if |z| > b
    Entries of :math:`x` smaller than a in magnitude are set to 0,
    entries with magnitudes larger than b are untouched,
    and entries in between have soft-thresholding applied.
    For simplicity we set :math:`b = 5*a` in this implementation.
    """
    upper_threshold = 5 * lower_threshold
    return prox_l0(x, upper_threshold) + prox_l1(x, lower_threshold) * (
        np.abs(x) < upper_threshold
    )


def get_prox(regularization):
    if regularization.lower() == "l0":
        return prox_l0
    elif regularization.lower() == "l1":
        return prox_l1
    elif regularization.lower() == "cad":
        return prox_cad
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))


def print_model(
    coef,
    input_features,
    errors=None,
    intercept=None,
    error_intercept=None,
    precision=3,
    pm="±",
):
    """
    Args:
        coef:
        input_features:
        errors:
        intercept:
        sigma_intercept:
        precision:
        pm:
    Returns:
    """

    def term(c, sigma, name):
        rounded_coef = np.round(c, precision)
        if rounded_coef == 0 and sigma is None:
            return ""
        elif sigma is None:
            return f"{c:.{precision}f} {name}"
        elif rounded_coef == 0 and np.round(sigma, precision) == 0:
            return ""
        else:
            return f"({c:.{precision}f} {pm} {sigma:.{precision}f}) {name}"

    errors = errors if errors is not None else repeat(None)
    components = [term(c, e, i) for c, e, i in zip(coef, errors, input_features)]
    eq = " + ".join(filter(bool, components))

    if not eq or intercept or error_intercept is not None:
        intercept = intercept or 0
        intercept_str = term(intercept, error_intercept, "").strip()
        if eq and intercept_str:
            eq += " + "
            eq += intercept_str
        elif not eq:
            eq = f"{intercept:.{precision}f}"
    return eq


def equations(pipeline, input_features=None, precision=3, input_fmt=None):
    input_features = pipeline.steps[0][1].get_feature_names(input_features)
    if input_fmt:
        input_features = [input_fmt(i) for i in input_features]
    coef = pipeline.steps[-1][1].coef_
    intercept = pipeline.steps[-1][1].intercept_
    if np.isscalar(intercept):
        intercept = intercept * np.ones(coef.shape[0])
    return [
        print_model(
            coef[i], input_features, intercept=intercept[i], precision=precision
        )
        for i in range(coef.shape[0])
    ]


def supports_multiple_targets(estimator):
    """Checkes whether estimator support mutliple targets."""
    if isinstance(estimator, MultiOutputMixin):
        return True
    try:
        return estimator._more_tags()["multioutput"]
    except (AttributeError, KeyError):
        return False

class BaseFeatureLibrary(TransformerMixin):
    """
    Base class for feature libraries.
    Forces subclasses to implement `fit`, `transform`,
    and `get_feature_names` functions.
    """

    def __init__(self, **kwargs):
        pass

    # Force subclasses to implement this
    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        Returns
        -------
        self : instance
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def transform(self, X):
        """
        Transform data.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.
        Returns
        -------
        XP : np.ndarray, [n_samples, n_output_features]
            The matrix of features, where n_output_features is the number
            of features generated from the combination of inputs.
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def get_feature_names(self, input_features=None):
        """Return feature names for output features.
        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.
        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        raise NotImplementedError

    @property
    def size(self):
        check_is_fitted(self)
        return self.n_output_features_


class CustomLibrary(BaseFeatureLibrary):
    """Generate a library with custom functions.
    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable.
    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list must be the
        same length as library_functions. If no list of function names is
        provided, defaults to using :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.
    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.
    function_names : list of functions
        Functions for generating string representations of each library
        function.
    n_input_features_ : int
        The total number of input features.
    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.
    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import CustomLibrary
    >>> X = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = CustomLibrary(library_functions=functions).fit(X)
    >>> lib.transform(X)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(self, library_functions, function_names=None, interaction_only=True):
        super(CustomLibrary, self).__init__()
        self.functions = library_functions
        self.function_names = function_names
        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        self.interaction_only = interaction_only

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.
        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.
        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )
        return feature_names

    def fit(self, X, y=None):
        """Compute number of output features.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(X).shape
        self.n_input_features_ = n_features
        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.functions)),
                )
            )
        return self

    def transform(self, X):
        """Transform data to custom features
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.
        Returns
        -------
        XP : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.
        """
        check_is_fitted(self)

        X = check_array(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        library_idx = 0
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                XP[:, library_idx] = f(*[X[:, j] for j in c])
                library_idx += 1

        return XP

class SINDy(BaseEstimator):
    """
    SINDy model object.
    Parameters
    ----------
    optimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be an object
        that extends the sindy.optimizers.BaseOptimizer class. Default is
        sequentially thresholded least squares with a threshold of 0.1.
    feature_library : feature library object, optional
        Default is polynomial features of degree 2.
    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object that
        extends the sindy.differentiation_methods.BaseDifferentiation class.
        Default is centered difference.
    feature_names : list of string, length n_input_features, optional
        Names for the input features. If None, will use ['x0','x1',...].
    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right hand side functions predict continuous time derivatives).
    n_jobs : int, optional (default 1)
        The number of parallel jobs to use when fitting, predicting with, and
        scoring the model.
    Attributes
    ----------
    model : sklearn.multioutput.MultiOutputRegressor object
        The fitted SINDy model.
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> model = SINDy()
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.000 1 + 10.000 x0
    x1' = 27.993 1 + -0.999 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    >>> model.coefficients()
    array([[ 0.        ,  0.        ,  0.        ],
           [-9.99969193, 27.99344519,  0.        ],
           [ 9.99961547, -0.99905338,  0.        ],
           [ 0.        ,  0.        , -2.66645651],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.99990257],
           [ 0.        , -0.99980268,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])
    >>> model.score(x, t=t[1]-t[0])
    0.999999985520653
    """

    def __init__(
        self,
        optimizer=None,
        feature_library=None,
        feature_names=None,
        discrete_time=False,
        n_jobs=1,
    ):
        if optimizer is None:
            optimizer = STLSQ()
        self.optimizer = optimizer
        if feature_library is None:
            feature_library = PolynomialFeatures()
        self.feature_library = feature_library
        self.feature_names = feature_names
        self.discrete_time = discrete_time
        self.n_jobs = n_jobs

    def fit(self, x, t, x_dot, multiple_trajectories=False, unbias=True, quiet=False):
        """
        Fit the SINDy model.
        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.
        t: float, numpy array of shape [n_samples], or list of numpy arrays, optional \
                (default 1)
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.
            Default value is a timestep of 1 between samples.
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features), \
                optional (default None)
            Optional pre-computed derivatives of the training data. If not
            provided, the time derivatives of the training data will be
            computed using the specified differentiation method. If x_dot is
            provided, it must match the shape of the training data and these
            values will be used as the time derivatives.
        multiple_trajectories: boolean, optional, (default False)
            Whether or not the training data includes multiple trajectories. If
            True, the training data must be a list of arrays containing data
            for each trajectory. If False, the training data must be a single
            array.
        unbias: boolean, optional (default True)
            Whether to perform an extra step of unregularized linear regression to
            unbias the coefficients for the identified support.
            If the optimizer (`SINDy.optimizer`) applies any type of regularization,
            that regularization may bias coefficients toward particular values,
            improving the conditioning of the problem but harming the quality of the
            fit. Setting `unbias=True` enables an extra step wherein unregularized
            linear regression is applied, but only for the coefficients in the support
            identified by the optimizer. This helps to remove the bias introduced by
            regularization.
        quiet: boolean, optional (default False)
            Whether or not to suppress warnings during model fitting.
        Returns
        -------
        self: returns an instance of self
        """
        

        x = validate_input(x, t)

        if self.discrete_time:
            x_dot = validate_input(x_dot)
        else:
            x_dot = validate_input(x_dot, t)

        # Drop rows where derivative isn't known
        x, x_dot = drop_nan_rows(x, x_dot)

        optimizer = self.optimizer
        steps = [("features", self.feature_library), ("model", optimizer)]
        self.model = Pipeline(steps)

        action = "ignore" if quiet else "default"
        with warnings.catch_warnings():
            warnings.filterwarnings(action, category=ConvergenceWarning)
            warnings.filterwarnings(action, category=LinAlgWarning)
            warnings.filterwarnings(action, category=UserWarning)

            self.model.fit(x, x_dot)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_

        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_input_features_):
                feature_names.append("x" + str(i))
            self.feature_names = feature_names

        return self

    def predict(self, x, multiple_trajectories=False):
        """
        Predict the time derivatives using the SINDy model.
        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples.
        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.
        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Predicted time derivatives
        """
        check_is_fitted(self, "model")
        if multiple_trajectories:
            x = [validate_input(xi) for xi in x]
            return [self.model.predict(xi) for xi in x]
        else:
            x = validate_input(x)
            return self.model.predict(x)

    def equations(self, precision=3):
        """
        Get the right hand sides of the SINDy model equations.
        Parameters
        ----------
        precision: int, optional (default 3)
            Number of decimal points to print for each coefficient in the
            equation.
        Returns
        -------
        equations: list of strings
            Strings containing the SINDy model equation for each input feature.
        """
        check_is_fitted(self, "model")
        if self.discrete_time:
            base_feature_names = [f + "[k]" for f in self.feature_names]
        else:
            base_feature_names = self.feature_names
        return equations(
            self.model, input_features=base_feature_names, precision=precision
        )

    def print(self, lhs=None, precision=3):
        """Print the SINDy model equations.
        Parameters
        ----------
        lhs: list of strings, optional (default None)
            List of variables to print on the left-hand sides of the learned equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        """
        eqns = self.equations(precision)
        for i, eqn in enumerate(eqns):
            if self.discrete_time:
                print(self.feature_names[i] + "[k+1] = " + eqn)
            elif lhs is None:
                print(self.feature_names[i] + "' = " + eqn)
            else:
                print(lhs[i] + " = " + eqn)

    def score(
        self,
        x,
        t=1,
        x_dot=None,
        multiple_trajectories=False,
        metric=r2_score,
        **metric_kws
    ):
        """
        Returns a score for the time derivative prediction.
        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples
        t: float, numpy array of shape [n_samples], or list of numpy arrays, optional \
                (default 1)
            Time step between samples or array of collection times. Optional,
            used to compute the time derivatives of the samples if x_dot is not
            provided.
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features), \
                optional
            Optional pre-computed derivatives of the samples. If provided,
            these values will be used to compute the score. If not provided,
            the time derivatives of the training data will be computed using
            the specified differentiation method.
        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.
        metric: metric function, optional
            Metric function with which to score the prediction. Default is the
            coefficient of determination R^2.
        metric_kws: dict, optional
            Optional keyword arguments to pass to the metric function.
        Returns
        -------
        score: float
            Metric function value for the model prediction of x_dot.
        """

        x = validate_input(x, t)
          
        if ndim(x_dot) == 1:
            x_dot = x_dot.reshape(-1, 1)

        # Drop rows where derivative isn't known (usually endpoints)
        x, x_dot = drop_nan_rows(x, x_dot)

        x_dot_predict = self.model.predict(x)
        return metric(x_dot_predict, x_dot, **metric_kws)

    def coefficients(self):
        """Return a list of the coefficients learned by SINDy model.
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_

    def get_feature_names(self):
        """Return a list of names of features used by SINDy model.
        """
        check_is_fitted(self, "model")
        return self.model.steps[0][1].get_feature_names(
            input_features=self.feature_names
        )

    def simulate(self, x0, t, integrator=odeint, stop_condition=None, **integrator_kws):
        """
        Simulate the SINDy model forward in time.
        Parameters
        ----------
        x0: numpy array, size [n_features]
            Initial condition from which to simulate.
        t: int or numpy array of size [n_samples]
            If the model is in continuous time, t must be an array of time
            points at which to simulate. If the model is in discrete time,
            t must be an integer indicating how many steps to predict.
        integrator: function object, optional
            Function to use to integrate the system. Default is scipy's odeint.
        stop_condition: function object, optional
            If model is in discrete time, optional function that gives a
            stopping condition for stepping the simulation forward.
        integrator_kws: dict, optional
            Optional keyword arguments to pass to the integrator
        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Simulation results
        """

        if self.discrete_time:
            if not isinstance(t, int):
                raise ValueError(
                    "For discrete time model, t must be an integer (indicating"
                    "the number of steps to predict)"
                )

            x = zeros((t, self.n_input_features_))
            x[0] = x0
            for i in range(1, t):
                x[i] = self.predict(x[i - 1 : i])
                if stop_condition is not None and stop_condition(x[i]):
                    return x[: i + 1]
            return x
        else:
            if isscalar(t):
                raise ValueError(
                    "For continuous time model, t must be an array of time"
                    " points at which to simulate"
                )

            def rhs(x, t):
                return self.predict(x[newaxis, :])[0]

            return integrator(rhs, x0, t, **integrator_kws)

    @property
    def complexity(self):
        return self.model.steps[-1][1].complexity

def LeapfrogSINDy(z,h,f):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

    dim = int(len(z)/2)

    z[dim:] = z[dim:]+h/2*f(z)
    z[:dim] = z[:dim]+h*z[dim:]
    z[dim:] = z[dim:]+h/2*f(z)

    return z
    
def gen_one_trajSINDy(traj_len,start,h,f1, f2 = None,n_h = 800):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,n_h+1):
      if f2 is None: 
        x=LeapfrogSINDy(x,h_gen,f1)
      else: 
        x=np.expand_dims(classicInt(x,f1,f2,h,verbose = False), 0).transpose()
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]
