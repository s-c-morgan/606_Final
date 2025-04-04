
#!/usr/bin/env python

# Library: pyLAR
#
# Copyright 2014 Kitware Inc. 28 Corporate Drive,
# Clifton Park, NY, 12065, USA.
#
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__license__   = "Apache License, Version 2.0"
__author__    = "Roland Kwitt, Kitware Inc., 2013"
__email__     = "E-Mail: roland.kwitt@kitware.com"
__status__    = "Development"

import numpy as np

class OutlierPursuit:
    """
    OutlierPursuit algorithm for robust principal component analysis.
    
    This class implements the Outlier Pursuit algorithm to decompose a matrix
    into a low-rank component and a column-sparse component.
    
    Parameters
    ----------
    gamma : float, optional (default=None)
        The regularization parameter controlling the trade-off between
        the low-rank approximation and the column-sparse component.
        
    tol : float, optional (default=1e-6)
        Tolerance for stopping criterion.
        
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
        
    eta : float, optional (default=0.9)
        Parameter for updating the step size.
        
    delta : float, optional (default=1e-5)
        Minimum relative value for the step size.
    
    Attributes
    ----------
    L_ : numpy.ndarray
        The low-rank component.
        
    C_ : numpy.ndarray
        The column-sparse component.
        
    n_iter_ : int
        Number of iterations needed for convergence.
        
    termination_criterion_ : float
        The value of the termination criterion at the end of optimization.
    """
    
    def __init__(self, gamma=None, tol=1e-6, max_iter=1000, eta=0.9, delta=1e-5):
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.eta = eta
        self.delta = delta
        
        # Results
        self.L_ = None
        self.C_ = None
        self.n_iter_ = None
        self.termination_criterion_ = None
    
    def _shrink_singular_values(self, matrix, threshold):
        """
        Soft thresholding of singular values (nuclear norm shrinkage).
        
        Parameters
        ----------
        matrix : numpy.ndarray
            Input matrix.
            
        threshold : float
            Shrinkage threshold.
            
        Returns
        -------
        numpy.ndarray
            Matrix after singular value thresholding.
        """
        U, S, V = np.linalg.svd(matrix, full_matrices=False)
        
        # Apply soft thresholding to singular values
        S_thresholded = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > threshold:
                S_thresholded[i] = S[i] - threshold
            elif S[i] < -threshold:
                S_thresholded[i] = S[i] + threshold
            # else: leave as zero
        
        # Reconstruct the matrix
        return np.dot(U * S_thresholded, V)
    
    def _shrink_columns(self, matrix, threshold):
        """
        Column-wise soft thresholding (l2,1 norm shrinkage).
        
        Parameters
        ----------
        matrix : numpy.ndarray
            Input matrix.
            
        threshold : float
            Shrinkage threshold.
            
        Returns
        -------
        numpy.ndarray
            Matrix after column thresholding.
        """
        m, n = matrix.shape
        output = np.zeros_like(matrix)
        
        for i in range(n):
            column = matrix[:, i]
            column_norm = np.linalg.norm(column, ord=2)
            
            if column_norm > threshold:
                # Apply soft thresholding to the column
                output[:, i] = column * (1 - threshold/column_norm)
            # else: leave column as zeros
            
        return output
    
    def fit(self, X, mask=None):
        """
        Fit the OutlierPursuit model to the data.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_features, n_samples)
            Input data matrix to be decomposed.
            
        mask : numpy.ndarray, shape (n_features, n_samples), optional (default=None)
            Binary matrix where 1 indicates observed entries and 0 indicates missing entries.
            If None, all entries are assumed to be observed.
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.gamma is None:
            raise ValueError("Gamma parameter must be specified")
            
        # If no mask is given, assume all entries are observed
        if mask is None:
            mask = np.ones(X.shape)
            
        # Initialize variables
        m, n = X.shape
        L_current = np.zeros(X.shape)
        C_current = np.zeros(X.shape)
        L_previous = np.zeros(X.shape)
        C_previous = np.zeros(X.shape)
        
        # Initialize step size and other parameters
        t_previous = 1
        t_current = 1
        mu_current = 0.99 * np.linalg.norm(X, ord=2)  # Initial step size
        mu_bar = self.delta * mu_current  # Minimum step size
        
        # Tolerance for stopping criterion based on the input matrix
        tolerance = self.tol * np.linalg.norm(X, 'fro')
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update extrapolation variables
            Y_L = L_current + ((t_previous - 1) / t_current) * (L_current - L_previous)
            Y_C = C_current + ((t_previous - 1) / t_current) * (C_current - C_previous)
            
            # Compute residual, applying mask for observed entries only
            residual = np.multiply((Y_L + Y_C - X), mask)
            
            # Gradient descent step
            G_L = Y_L - 0.5 * residual
            G_C = Y_C - 0.5 * residual
            
            # Proximal operators
            L_new = self._shrink_singular_values(G_L, mu_current / 2)
            C_new = self._shrink_columns(G_C, mu_current * self.gamma / 2)
            
            # Update parameters for next iteration
            t_new = (1 + np.sqrt(4 * t_current**2 + 1)) / 2
            mu_new = max(self.eta * mu_current, mu_bar)
            
            # Calculate termination criterion
            S_L = 2 * (Y_L - L_new) + (L_new + C_new - Y_L - Y_C)
            S_C = 2 * (Y_C - C_new) + (L_new + C_new - Y_L - Y_C)
            
            term_criterion = (np.linalg.norm(S_L, 'fro')**2 + 
                              np.linalg.norm(S_C, 'fro')**2)
            
            # Check convergence
            if term_criterion <= tolerance**2:
                break
                
            # Update variables for next iteration
            L_previous = L_current
            L_current = L_new
            C_previous = C_current
            C_current = C_new
            t_previous = t_current
            t_current = t_new
            mu_current = mu_new
        
        # Store final results
        self.L_ = L_new
        self.C_ = C_new
        self.n_iter_ = iteration + 1
        self.termination_criterion_ = term_criterion
        
        return self
    
    def fit_transform(self, X, mask=None):
        """
        Fit the model to the data and return the low-rank component.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_features, n_samples)
            Input data matrix to be decomposed.
            
        mask : numpy.ndarray, shape (n_features, n_samples), optional (default=None)
            Binary matrix where 1 indicates observed entries and 0 indicates missing entries.
            If None, all entries are assumed to be observed.
            
        Returns
        -------
        L : numpy.ndarray, shape (n_features, n_samples)
            The low-rank component.
        """
        self.fit(X, mask)
        return self.L_
    
    def transform(self, X, mask=None):
        """
        Apply the fitted model to new data.
        
        This is not typically used for OutlierPursuit, but is provided for
        scikit-learn compatibility.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_features, n_samples)
            New data matrix.
            
        mask : numpy.ndarray, shape (n_features, n_samples), optional (default=None)
            Binary matrix for the new data.
            
        Returns
        -------
        X_transformed : numpy.ndarray, shape (n_features, n_samples)
            X with outlier columns removed.
        """
        if self.L_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Simply run the algorithm on the new data
        model_copy = OutlierPursuit(
            gamma=self.gamma,
            tol=self.tol,
            max_iter=self.max_iter,
            eta=self.eta,
            delta=self.delta
        )
        model_copy.fit(X, mask)
        
        return model_copy.L_
    
    def get_outlier_columns(self, threshold=1e-6):
        """
        Identify outlier columns based on the column-sparse component.
        
        Parameters
        ----------
        threshold : float, optional (default=1e-6)
            Columns with norms above this threshold are considered outliers.
            
        Returns
        -------
        outliers : numpy.ndarray
            Boolean array indicating which columns are outliers.
        """
        if self.C_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        column_norms = np.linalg.norm(self.C_, axis=0)
        return column_norms > threshold