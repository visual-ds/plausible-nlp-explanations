import autograd.numpy as np
import numbers

from autograd import elementwise_grad as egrad
from scipy import optimize
from joblib import Parallel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _check_solver, _check_multi_class
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import _check_sample_weight

_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)

# scikit-learn's BSD 3-Clause License

"""
BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

def logsumexp(p, axis):
    """Compute the log of the sum of exponentials of input elements.

    Args:
        p (array-like): Input array.
        axis (int): Along which axis the sum is taken.

    Returns:
        array-like: The result of the log of the sum of exponentials of the
            input array elements.
    """
    p_max = np.amax(p, axis=axis, keepdims=True)  # Trick to improve numeric computation
    p = p - p_max
    p = np.exp(p)
    p = np.sum(p, axis=axis)
    p = np.log(p)
    p = p + np.squeeze(p_max, axis)
    return p

def _multinomial_loss(w, X, Y, X_ratio, Y_ratio, w1, w2, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray of shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = np.matmul(X, w.T)
    p = p + intercept
    p = p - logsumexp(p, axis=1)[:, np.newaxis]
    loss = -np.sum(sample_weight * Y * p)

    # Contrastive
    p_ratio = np.dot(X_ratio, w.T)
    p_ratio = p_ratio + intercept
    if X_ratio.shape[1] == 1:
        # If there is only the positive rationale,
        # then a "zero rationale" is added as a negative rationale to simulate
        # the 1 in the denominator of the sigmoid function.
        # Instead of adding a vector of zeros to the `X_ratio` tensor, we add
        # a vector of zeros to `p_ratio` to ignore the `intercept`.
        p_ratio = np.concatenate((p_ratio, np.zeros_like(p_ratio)), axis=1)
    p_ratio = p_ratio[:, 0, :] - logsumexp(p_ratio, axis=1)
    loss_ratio = -np.sum(Y_ratio*p_ratio)

    loss = w1*loss + w2*loss_ratio
    w_ravel = w.ravel()
    loss = loss + 0.5 * alpha * np.dot(w_ravel, w_ravel)

    return loss

def _multinomial_loss_grad(w, X, Y, X_ratio, Y_ratio, w1, w2, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray of shape (n_classes * n_features,) or \
            (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    def _multinomial_loss_w(x):
        return _multinomial_loss(x, X, Y, X_ratio, Y_ratio, w1, w2, alpha, sample_weight)

    _multinomial_loss_w_grad = egrad(_multinomial_loss_w)

    loss = _multinomial_loss_w(w)
    grad = _multinomial_loss_w_grad(w)
    return loss, grad, None

def _logistic_regression_path(
    X,
    y,
    X_ratio,
    y_ratio,
    w1,
    w2,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    # solver="lbfgs",
    coef=None,
    class_weight=None,
    # dual=False,
    # penalty="l2",
    intercept_scaling=1.0,
    # multi_class="multinomial",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    # l1_ratio=None,
):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or array-like of shape (n_cs,), default=10
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Numerical solver to use.

    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    solver = "lbfgs"
    dual = False
    penalty = "l2"
    multi_class = "multinomial"
    l1_ratio = None

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    # if pos_class is None and multi_class != "multinomial":
    #     if classes.size > 2:
    #         raise ValueError("To fit OvR, use the pos_class argument")
    #     # np.unique(y) gives labels in sorted order.
    #     pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    # if isinstance(class_weight, dict) or multi_class == "multinomial":
    class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
    sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    # if multi_class == "ovr":
    #     w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    #     mask_classes = np.array([-1, 1])
    #     mask = y == pos_class
    #     y_bin = np.ones(y.shape, dtype=X.dtype)
    #     y_bin[~mask] = -1.0
    #     # for compute_class_weight

    #     if class_weight == "balanced":
    #         class_weight_ = compute_class_weight(
    #             class_weight, classes=mask_classes, y=y_bin
    #         )
    #         sample_weight *= class_weight_[le.fit_transform(y_bin)]

    # else:
    # if solver not in ["sag", "saga"]:
    lbin = LabelBinarizer()
    Y_multi = lbin.fit_transform(y)
    Y_multi_ratio = lbin.transform(y_ratio)
    if Y_multi.shape[1] == 1:
        Y_multi = np.hstack([1 - Y_multi, Y_multi])
        Y_multi_ratio = np.hstack([1 - Y_multi_ratio, Y_multi_ratio])
    # else:
    #     # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
    #     le = LabelEncoder()
    #     Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

    w0 = np.zeros(
        (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
    )

    if coef is not None:
        # it must work both giving the bias term and not
        # if multi_class == "ovr":
        #     if coef.size not in (n_features, w0.size):
        #         raise ValueError(
        #             "Initialization coef is of shape %d, expected shape %d or %d"
        #             % (coef.size, n_features, w0.size)
        #         )
        #     w0[: coef.size] = coef
        # else:
        # For binary problems coef.shape[0] should be 1, otherwise it
        # should be classes.size.
        n_classes = classes.size
        # if n_classes == 2:
        #     n_classes = 1

        if coef.shape[0] != n_classes or coef.shape[1] not in (
            n_features,
            n_features + 1,
        ):
            raise ValueError(
                "Initialization coef is of shape (%d, %d), expected "
                "shape (%d, %d) or (%d, %d)"
                % (
                    coef.shape[0],
                    coef.shape[1],
                    classes.size,
                    n_features,
                    classes.size,
                    n_features + 1,
                )
            )

        # if n_classes == 1:
        #     w0[0, : coef.shape[1]] = -coef
        #     w0[1, : coef.shape[1]] = coef
        # else:
        w0[:, : coef.shape[1]] = coef

    # if multi_class == "multinomial":
    # scipy.optimize.minimize and newton-cg accepts only
    # ravelled parameters.
    # if solver in ["lbfgs", "newton-cg"]:
    w0 = w0.ravel()
    target = Y_multi
    target_ratio = Y_multi_ratio
    # if solver == "lbfgs":

    def func(x, *args):
        return _multinomial_loss_grad(x, *args)[0:2]

    # elif solver == "newton-cg":

    #     def func(x, *args):
    #         return _multinomial_loss(x, *args)[0]

    #     def grad(x, *args):
    #         return _multinomial_loss_grad(x, *args)[1]

    #     hess = _multinomial_grad_hess
    warm_start_sag = {"coef": w0.T}
    # else:
    #     target = y_bin
    #     if solver == "lbfgs":
    #         func = _logistic_loss_and_grad
    #     elif solver == "newton-cg":
    #         func = _logistic_loss

    #         def grad(x, *args):
    #             return _logistic_loss_and_grad(x, *args)[1]

    #         hess = _logistic_grad_hess
    #     warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        # if solver == "lbfgs":
        iprint = [-1, 50, 1, 100, 101][
            np.searchsorted(np.array([0, 1, 2, 3]), verbose)
        ]
        opt_res = optimize.minimize(
            func,
            w0,
            method="L-BFGS-B",
            jac=True,
            args=(X, target, X_ratio, target_ratio, w1, w2, 1.0 / C, sample_weight),
            options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
        )
        n_iter_i = _check_optimize_result(
            solver,
            opt_res,
            max_iter,
            extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
        )
        w0, loss = opt_res.x, opt_res.fun
        # elif solver == "newton-cg":
        #     args = (X, target, 1.0 / C, sample_weight)
        #     w0, n_iter_i = _newton_cg(
        #         hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
        #     )
        # elif solver == "liblinear":
        #     coef_, intercept_, n_iter_i, = _fit_liblinear(
        #         X,
        #         target,
        #         C,
        #         fit_intercept,
        #         intercept_scaling,
        #         None,
        #         penalty,
        #         dual,
        #         verbose,
        #         max_iter,
        #         tol,
        #         random_state,
        #         sample_weight=sample_weight,
        #     )
        #     if fit_intercept:
        #         w0 = np.concatenate([coef_.ravel(), intercept_])
        #     else:
        #         w0 = coef_.ravel()

        # elif solver in ["sag", "saga"]:
        #     if multi_class == "multinomial":
        #         target = target.astype(X.dtype, copy=False)
        #         loss = "multinomial"
        #     else:
        #         loss = "log"
        #     # alpha is for L2-norm, beta is for L1-norm
        #     if penalty == "l1":
        #         alpha = 0.0
        #         beta = 1.0 / C
        #     elif penalty == "l2":
        #         alpha = 1.0 / C
        #         beta = 0.0
        #     else:  # Elastic-Net penalty
        #         alpha = (1.0 / C) * (1 - l1_ratio)
        #         beta = (1.0 / C) * l1_ratio

        #     w0, n_iter_i, warm_start_sag = sag_solver(
        #         X,
        #         target,
        #         sample_weight,
        #         loss,
        #         alpha,
        #         beta,
        #         max_iter,
        #         tol,
        #         verbose,
        #         random_state,
        #         False,
        #         max_squared_sum,
        #         warm_start_sag,
        #         is_saga=(solver == "saga"),
        #     )

        # else:
        #     raise ValueError(
        #         "solver must be one of {'liblinear', 'lbfgs', "
        #         "'newton-cg', 'sag'}, got '%s' instead" % solver
        #     )

        # if multi_class == "multinomial":
        n_classes = max(2, classes.size)
        multi_w0 = np.reshape(w0, (n_classes, -1))
        # if n_classes == 2:
        #     assert np.allclose(-multi_w0[0], multi_w0[1])
        #     multi_w0 = multi_w0[1][np.newaxis, :]
        coefs.append(multi_w0.copy())
        # else:
        #     coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter

class ContrastiveLogisticRegression(LogisticRegression):
    def __init__(
        self,
        *,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    ):
        super().__init__(
            penalty='l2',
            dual=False,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver='lbfgs',
            max_iter=max_iter,
            multi_class='multinomial',
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=None
        )

    def fit(self, X, y, X_ratio, y_ratio, w1, w2, sample_weight=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        The SAGA solver supports both float64 and float32 bit arrays.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        # if self.penalty == "elasticnet":
        #     if (
        #         not isinstance(self.l1_ratio, numbers.Number)
        #         or self.l1_ratio < 0
        #         or self.l1_ratio > 1
        #     ):
        #         raise ValueError(
        #             "l1_ratio must be between 0 and 1; got (l1_ratio=%r)"
        #             % self.l1_ratio
        #         )
        # elif self.l1_ratio is not None:
        #     warnings.warn(
        #         "l1_ratio parameter is only used when penalty is "
        #         "'elasticnet'. Got "
        #         "(penalty={})".format(self.penalty)
        #     )
        # if self.penalty == "none":
        #     if self.C != 1.0:  # default values
        #         warnings.warn(
        #             "Setting penalty='none' will ignore the C and l1_ratio parameters"
        #         )
        #         # Note that check for l1_ratio is done right above
        #     C_ = np.inf
        #     penalty = "l2"
        # else:
        C_ = self.C
        penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        # if solver == "lbfgs":
        _dtype = np.float64
        # else:
        #     _dtype = [np.float64, np.float32]

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        # if solver == "liblinear":
        #     if effective_n_jobs(self.n_jobs) != 1:
        #         warnings.warn(
        #             "'n_jobs' > 1 does not have any effect when"
        #             " 'solver' is set to 'liblinear'. Got 'n_jobs'"
        #             " = {}.".format(effective_n_jobs(self.n_jobs))
        #         )
        #     self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
        #         X,
        #         y,
        #         self.C,
        #         self.fit_intercept,
        #         self.intercept_scaling,
        #         self.class_weight,
        #         self.penalty,
        #         self.dual,
        #         self.verbose,
        #         self.max_iter,
        #         self.tol,
        #         self.random_state,
        #         sample_weight=sample_weight,
        #     )
        #     self.n_iter_ = np.array([n_iter_])
        #     return self

        # if solver in ["sag", "saga"]:
        #     max_squared_sum = row_norms(X, squared=True).max()
        # else:
        max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )

        # if len(self.classes_) == 2:
        #     n_classes = 1
            # classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
            )

        # Hack so that we iterate only once for the multinomial case.
        # if multi_class == "multinomial":
        classes_ = [None]
        warm_start_coef = [warm_start_coef]
        # if warm_start_coef is None:
        #     warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        # if solver in ["sag", "saga"]:
        #     prefer = "threads"
        # else:
        prefer = "processes"
        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
                y,
                X_ratio,
                y_ratio,
                w1,
                w2,
                pos_class=class_,
                Cs=[C_],
                # l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                verbose=self.verbose,
                # solver=solver,
                # multi_class=multi_class,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                check_input=False,
                random_state=self.random_state,
                coef=warm_start_coef_,
                # penalty=penalty,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
            )
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
        )

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        n_features = X.shape[1]
        # if multi_class == "multinomial":
        self.coef_ = fold_coefs_[0][0]
        # else:
        #     self.coef_ = np.asarray(fold_coefs_)
        #     self.coef_ = self.coef_.reshape(
        #         n_classes, n_features + int(self.fit_intercept)
        #     )

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)

        return self
    
    def losses(self, X, y, X_ratio, y_ratio):
        coef = self.coef_
        if self.fit_intercept:
            intercept = self.intercept_
            w = np.hstack([coef, intercept[..., np.newaxis]]).ravel()
        else:
            w = coef.ravel()
        lbin = LabelBinarizer()
        Y = lbin.fit_transform(y)
        # Important to note that y may not have all labels. In our current use
        # of it, this is not a problem, but it could be in the future.
        Y_ratio = lbin.transform(y_ratio)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])
            Y_ratio = np.hstack([1 - Y_ratio, Y_ratio])
        sample_weight = np.ones(X.shape[0])
        loss = _multinomial_loss(w, X, Y, X_ratio, Y_ratio, 1, 0, 1.0/self.C, sample_weight)
        loss_ratio = _multinomial_loss(w, X, Y, X_ratio, Y_ratio, 0, 1, 1.0/self.C, sample_weight)
        return np.array([loss, loss_ratio])
