"""
Created by Tetsu Haruyama
"""

import numpy as np
from scipy.stats import truncnorm
import statsmodels.api as sm
from py4etrics.base_for_models import GenericLikelihoodModel_TobitTruncreg

class Truncreg(GenericLikelihoodModel_TobitTruncreg):
    """
    Method 1:
    Truncreg(endog, exog, left=<-np.inf>, right=<np.inf>).fit()
    endog = dependent variable
    exog = independent variable (add constant if needed)
    left = the threshold value for left-truncation (default:-np.inf)
    right = the threshold value for right-truncation (default:np.inf)

    Method 2:
    formula = 'y ~ 1 + x'
    Truncreg(formula, left=<-np.inf>, right=<np.inf>, data=<DATA>).fit()

    Note:
    Left-truncated Regression if 'left' only is set.
    Right-truncated Regression if 'right' only is set.
    Left- and Right-truncated Regression if 'left' and 'right' both are set.

    """

    def __init__(self, endog, exog, left=None, right=None, **kwds):
        super(Truncreg, self).__init__(endog, exog, **kwds)

        if left == None:
            left = -np.inf
        self.left = left

        if right == None:
            right = np.inf
        self.right = right

    def loglikeobs(self, params):
        s = params[-1]
        beta = params[:-1]

        def _truncreg(y, x, left, right, beta, s):
            Xb = np.dot(x, beta)
            _l = (left - Xb)/np.exp(s)
            _r = (right - Xb)/np.exp(s)
            return truncnorm.logpdf(y, a=_l, b=_r, loc=Xb, scale=np.exp(s))

        return _truncreg(self.endog, self.exog,
                         self.left, self.right, beta, s)


    def fit(self, cov_type='nonrobust', start_params=None, maxiter=10000, maxfun=10000, **kwds):
        # add sigma for summary
        if 'Log(Sigma)' not in self.exog_names:
            self.exog_names.append('Log(Sigma)')
        else:
            pass
        # initial guess
        res_ols = sm.OLS(self.endog, self.exog).fit()
        params_ols = res_ols.params
        sigma_ols = np.log(np.std(res_ols.resid))
        if start_params == None:
            start_params = np.append(params_ols, sigma_ols)

        return super(Truncreg, self).fit(cov_type=cov_type, start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun, **kwds)
    def get_expectation(self, at='all', atexog=None, expec_type='latent'):
        """Get the estimated expected value of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The expected value at each observation.
            - 'mean', The expected value at the mean of each regressor.
            - 'median', The expected value at the median of each regressor.
            - 'zero', The expected value at zero for each regressor.
            - 'all', The expected value at each observation. 

        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the expected value.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        expec_type : str, optional
            Options are:

            - 'latent', The expected value of the latent variable.
            - 'conditional', The expected value of the dependent variable given it is not truncated.
            - 'total', The expected value of the dependent variable. 
            Non observable values are replaced by left and right truncation values.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        """
        self._reset() # always reset the cache when this is called

        results = self.results
        model = results.model
        
        beta_hat = results.params[:-1]
        
        log_sigma_hat = results.params[-1]
        sigma_hat = np.exp(log_sigma_hat)

        exog = result.exog.copy() # copy because values are changed

        # TODO: here we update exog... 'all' without transformation

        expec_latent = exog @ beta_hat.T
        if expec_type == 'latent':
            return expec_latent

        expec_conditional = truncnorm.mean(a=self.left, b=self.right, loc=expec_latent, scale=sigma_hat)
        if expec_type == 'conditional':
            return expec_conditional

        if expec_type == 'total':
            right_truncation_prob = 1 - truncnorm.cdf(a=self.right, b=self.right, loc=expec_latent, scale=sigma_hat)
            left_truncation_prob = truncnorm.cdf(a=self.left, b=self.right, loc=expec_latent, scale=sigma_hat)
            no_truncation_prob = 1 - right_truncation_prob - left_truncation_prob

            left_expec_part = 0
            if self.left > -np.inf:
                left_expec_part = self.left * left_truncation_prob
            right_expec_part = 0
            if self.right < np.inf:
                right_expec_part = self.right * right_truncation_prob
            middle_expec_part = expec_conditional * no_truncation_prob
            return left_expec_part + middle_expec_part + right_expec_part
# EOF
