import numpy as np
from scipy import stats


class LinearRegression:
    """
    - Least squares approximation of mean (fit + predict)
    - d: number of features (excluding intercept)
    - n: sample size
    - sample variance (unbiased estimator): sigma^2_hat = SSE / (n - d - 1)
    - standard deviation: sigma_hat = sqrt(sigma^2_hat)
    - RMSE: sqrt(SSE / n)
    """

    def __init__(self, add_intercept: bool = True):        
        self.add_intercept = add_intercept # om True lägger vi till en kolumn med 1:or

        # Modellparametrar
        self.beta_ = None     # regressionskoefficienter (inkl intercept)
        self.n_ = None        # antal observationer
        self.d_ = None        # antal features (exkl intercept)
        self._X = None   # design matrix
        self._y = None   # response vector    
        # Felstorheter 
        self.sse_ = None     # Sum of Squared Errors
       

    def __repr__(self):
        if self.beta_ is None:
            return "LinearRegression(unfitted)" # Om modellen är inte tränad
        beta = self.beta_.reshape(-1)
        terms = [f"{b:.4f}" for b in beta]
    
        return "Y = " + " + ".join(
        [terms[0]] + [f"{terms[i]}*X{i}" for i in range(1, len(terms))]
        )
    def __str__(self):
        return self.__repr__()

    def _as_2d(self, X: np.ndarray):# Säkerställer att X alltid är en 2D-matris
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _design_matrix(self, X: np.ndarray): # bygger matrisen A
        X = self._as_2d(X)
        if self.add_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            return np.column_stack([ones, X.astype(float)])
        return X.astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray):#  Tränar modellen (hittar koefficienterna beta)
        """
        Fit OLS
        beta = pinv(A^T A) A^T y
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        A = self._design_matrix(X)

        if A.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows/observations.")
        # Spara träningsdata i objektet 
        self._X = A
        self._y = y 

        # spara n och d
        self.n_ = A.shape[0]
        self.d_ = A.shape[1] - (1 if self.add_intercept else 0)

        # OLS solution 
        self.beta_ = np.linalg.pinv(A.T @ A) @ (A.T @ y)

        # Beräkna SSE 
        y_hat = A @ self.beta_
        residuals = y - y_hat
        self.sse_ = float(np.sum(residuals ** 2))
        return self
    
    def predict(self, X: np.ndarray):# Gör prediktioner på ny data
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        A = self._design_matrix(X)
        return A @ self.beta_
    
    def residuals(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y, dtype=float).reshape(-1)
        return y - self.predict(X).reshape(-1)


    # --- Quantities required for G ---

    def sse(self):
        if self.sse_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self.sse_

    def sample_variance(self):
        """
        Unbiased estimator for multiple linear regression:
        sigma^2_hat = SSE / (n - d - 1)
        """
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        dof = self.n_ - self.d_ - 1
        if dof <= 0:
            raise ValueError("Not enough degrees of freedom to estimate variance.")
        return self.sse_ / dof

    def standard_deviation(self):
        return float(np.sqrt(self.sample_variance()))

    def rmse(self):
        """
        RMSE = sqrt(SSE / n)
        (Not unbiased, but required in PM)
        """
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return float(np.sqrt(self.sse_ / self.n_))
    
    #========== VG============
    def f_test(self, alpha=0.05):
        if self.beta_ is None:
           raise RuntimeError("Model is not fitted.")
 
        y = self._y.reshape(-1)
        n = self.n_
        d = self.d_

        # Syy
        y_mean = np.mean(y)
        Syy = np.sum((y - y_mean) ** 2)

        # SSR
        SSR = Syy - self.sse_

       # sigma^2_hat
        sigma2_hat = self.sse_ / (n - d - 1)

        # F-statistic
        F = (SSR / d) / sigma2_hat

        # p-value
        p_value = stats.f.sf(F, d, n - d - 1)

        reject = p_value < alpha

        return F, p_value, reject

    def r_squared(self): #     Reports how much of the variance in y is explained by the model.

        if self.beta_ is None:
          raise RuntimeError("Model is not fitted.")

        y = self._y.reshape(-1)
        y_mean = np.mean(y)
        Syy = np.sum((y - y_mean) ** 2)
        SSR = Syy - self.sse_
        return SSR / Syy
  
    def parameter_test(self, i, alpha=0.05):
         """
        Performs a t-test for a single parameter beta[i].
        Returns estimate, standard error, t-statistic, p-value and significance.
        """

         if  self.beta_ is None:
            raise RuntimeError("Model is not fitted.")

         X = self._X
         n = X.shape[0]
         p = X.shape[1]  # number of parameters (including intercept)

         if n - p <= 0:
            raise RuntimeError("Not enough degrees of freedom.")

         if i < 0 or i >= p:
             raise IndexError(f"Parameter index {i} is out of bounds.")

        # Estimate of residual variance
         sigma2_hat = self.sse_ / (n - p)
         S = np.sqrt(sigma2_hat)

       # Inverse of normal equation matrix XtX_inv
         c = np.linalg.pinv(X.T @ X)

         beta = self.beta_.reshape(-1)

         df = n - p  # degrees of freedom

        # Standard error for beta[i]
         se_i = S * np.sqrt(c[i, i])

       # t-statistic for H0: beta[i] = 0
         t_i = beta[i] / se_i

         # Two-sided p-value
         p_i = 2 * min(
          stats.t.cdf(t_i, df),
          stats.t.sf(t_i, df)
             )

         return {
        "i": i,
        "beta": beta[i],
        "se": se_i,
        "t": t_i,
        "p": p_i,
        "significant": p_i < alpha
        }

    def pearson_matrix(self): 
        """ 
        Pearson correlation between explanatory variables 
        (without intercept column) 
        """ 
        X = self._X[:, 1:] if self.add_intercept else self._X 

        return np.corrcoef(X, rowvar=False) 
