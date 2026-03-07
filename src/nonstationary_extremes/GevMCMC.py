import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import genextreme
from scipy.linalg import sqrtm
from scipy.special import logsumexp

from scipy.stats import genextreme as gev

class GevMCMC:

    def __init__(self, data, param_setup, upper_limit=None, verbose=True):

        """
        This function sets up the data and the model parameters and hyperparameters for fitting a GEV model.

        Input
        data: T x 3 DataFrame where each column denotes each scenario 
        """
        
        self.data = data.dropna()
        self.upper_limit = upper_limit
        self.param_setup = param_setup
        self.verbose = verbose
        self.nT = len(self.data.iloc[:, 0])
        self.Tim = np.linspace(0, 1, self.nT)
        self.nPrm, self.param_structure = self.setup_params()
        self.initial_params = self.find_starting_parameters()
    
    def setup_params(self):
        """
        This function sets up the correct parameters given a string. It sets up the total number of parameters, along with the parameter indices.

        Parameter arrangement is
         1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  (21)
         1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16      17      18      (18)
         1   2   3   4   5   6   7   8   9      10      11      12  13      14      15      (15)
         1   2       3       4       5   6       7       8       9  10      11      12      (12)
         1   2       3       4       5   6       7       8       9                          ( 9)
         1   2       3       4       5                           6                          ( 6)
         1                           2                           3                          ( 3)
        m0 m11 m12 m21 m22 m31 m32  s0 s11 s12 s21 s22 s31 s32  x0 x11 x12 x21 x22 x31 x32
        """

        config_map = {
        "CCC": (3, [0, 0, 0, 0]),       # Only constant terms
        "LCC": (6, [1, 0, 0, 0]),       # Linear Mu, constant Sgm and Xi
        "LLC": (9, [1, 1, 0, 0]),       # Linear Mu and Sgm, constant Xi
        "QCC": (9, [2, 0, 0, 0]),       # Quadratic Mu, constant Sgm and Xi
        "LLL": (12, [1, 1, 1, 0]),      # Linear for all
        "QLC": (12, [2, 1, 0, 0]),      # Quadratic Mu, Linear Sgm, constant Xi
        "QLL": (15, [2, 1, 1, 0]),      # Quadratic Mu, Linear Sgm and Xi
        "QQC": (15, [2, 2, 0, 0]),      # Quadratic Mu and Sgm, constant Xi
        "QQL": (18, [2, 2, 1, 0]),      # Quadratic Mu and Sgm, Linear Xi
        "QQQ": (21, [2, 2, 2, 0]),      # Quadratic MU, Sgm, Xi
        "ACC": (9, [2, 0, 0, 1])        # Asymptotic QCC model
        }

        if self.param_setup not in config_map:
            raise ValueError(f"Invalid parameter setup: {self.param_setup}")
        
        nPrm, param_structure = config_map[self.param_setup]

        return nPrm, param_structure
    
    def setup_dataframes(self):
        """
        Dynamically sets up the `samples` and `total_accepted` DataFrames 
        based on the parameter structure and number of scenarios.
        """
        column_names = []
        for param_type, param_name in zip(self.param_structure, ["mu", "s", "x"]):
            if param_type >= 0:  # Constant term
                column_names.append(f"{param_name}_0")
            for scenario in range(1, 4): # Loop through scenarios
                for degree in range(1, param_type + 1):  # Loop through degrees (1 = Linear, 2 = Quadratic)
                    column_names.append(f"{param_name}_{scenario}{degree}")

        # Create the DataFrames with the generated columns
        samples = pd.DataFrame(columns=column_names)
        total_accepted = pd.DataFrame(columns=column_names)

        return samples, total_accepted
    
    def sigmoid_mu(self, x, a, b, c):
        """
        Compute the proposed sigmoid-like reparameterisation:
        y = a + (b) * (1 - exp(-x*(c+0.5))) / (1 + exp(-x*(c+0.5)))(1 + exp(-(c+0.5))) / (1 - exp(-(c+0.5)))
        x : array-like (nT,)
        a : scalar
        b : scalar
        c : scalar (should be > 0 ideally)
        Returns array shape (nT,)
        """

        # compute kernel
        z = np.exp(-np.asarray(x) * (c+0.5))
        z_end = np.exp(-1 * (c+0.5))
        kernel = (1 - z) / (1 + z)
        scale = (b)*((1+z_end) / (1-z_end))
        return a + scale * kernel
    
    def build_parameter_arrays(self, params, time_steps):
        """
        Build Mu, Sgm, Xi arrays for 3 scenarios across time_steps.

        - param_structure: list/tuple with degrees for [Mu, Sgm, Xi] where degree is 0 (const), 1 (linear), 2 (quadratic)
        - params: 1D array-like containing the parameters in the same relative ordering you used originally
        - time_steps: array-like of length nT (e.g. Tim)

        NOTE: In 'sigmoid' mode the Mu block of params is expected to contain:
            params[Mu_start] -> a (shared)
            params[Mu_start+1:Mu_start+4] -> b1,b2,b3
            params[Mu_start+4:Mu_start+7] -> c1,c2,c3
        """

        params = np.asarray(params)
        time_steps = np.asarray(time_steps)
        nT = time_steps.size

        # prepare outputs with shape (nT, 3)
        Mu = np.zeros((nT, 3))
        Sgm = np.zeros((nT, 3))
        Xi = np.zeros((nT, 3))

        # compute starting indices for each parameter block
        Mu_start = 0
        Sgm_start = Mu_start + 3 * self.param_structure[0] + 1
        Xi_start = Sgm_start + 3 * self.param_structure[1] + 1

        for iS in range(3):
            # MU block
            if self.param_structure[3] == 0:
                # original constant/linear/quadratic mapping
                Mu[:, iS] = params[Mu_start]  # constant term
                if self.param_structure[0] >= 1:  # linear
                    Mu[:, iS] += params[Mu_start + self.param_structure[0]*iS + 1] * time_steps
                if self.param_structure[0] >= 2:  # quadratic
                    Mu[:, iS] += params[Mu_start + self.param_structure[0]*iS + 2] * (time_steps ** 2)
            elif self.param_structure[3] == 1:

                if self.param_structure[0] != 2:
                    raise ValueError
                
                # Using new parameter mapping: a + (b/2)*(1-exp(-x/c))/(1+exp(-x/c))
                a = params[Mu_start]
                b = params[Mu_start + 2*iS + 1]
                c = params[Mu_start + 2*iS + 2]

                Mu[:, iS] = self.sigmoid_mu(time_steps, a, b, c)

            # Sgm block
            Sgm[:, iS] = params[Sgm_start]  # constant term
            if self.param_structure[1] >= 1:  # Linear term
                Sgm[:, iS] += params[Sgm_start + self.param_structure[1]*iS + 1] * time_steps
            if self.param_structure[1] >= 2:  # Quadratic term
                Sgm[:, iS] += params[Sgm_start + self.param_structure[1]*iS + 2] * (time_steps ** 2)

            # Xi block
            Xi[:, iS] = params[Xi_start]  # constant term
            if self.param_structure[2] >= 1:  # Linear term
                Xi[:, iS] += params[Xi_start + self.param_structure[2]*iS + 1] * time_steps
            if self.param_structure[2] >= 2:  # Quadratic term
                Xi[:, iS] += params[Xi_start + self.param_structure[2]*iS + 2] * (time_steps ** 2)

        return Mu, Sgm, Xi
        
    def log_likelihood(self, params, time_steps, data=None, return_pointwise=False):
        """
        Compute the (negative) log-likelihood for the GEV distribution.

        Parameters
        ----------
        params : dict or array-like
            Current parameter values (mu, sigma, xi) used to compute the likelihood.
        time_steps : array-like
            The time indices (or covariate time steps) used to build parameter arrays.
        data : pd.DataFrame or np.ndarray, optional
            The data to evaluate the likelihood on.
            If None, defaults to self.data (training data).
        return_pointwise : bool, optional
            If True, also return per-observation log-pdf values.

        Returns
        -------
        total_nll : float
            Total negative log-likelihood (scalar).
        tNll : np.ndarray
            Per-site total NLL (shape depends on data).
        log_pdf : np.ndarray, optional
            Individual log-pdf values (if return_pointwise=True).
        """

        # Allow custom dataset
        if data is None:
            data_np = self.data.values
        elif isinstance(data, pd.DataFrame):
            data_np = data.values
        else:
            data_np = np.asarray(data)
        
        # Build mu, sigma, xi arrays for the provided time steps
        Mu, Sgm, Xi = self.build_parameter_arrays(params, time_steps)  # shapes match data_np

        # Standardise
        z = (data_np - Mu) / Sgm
        t = 1 + Xi * z

        # Check domain validity (GEV constraint: t > 0, sigma > 0)
        invalid_mask = (t <= 0) | (Sgm <= 0)
        if np.any(invalid_mask):
            if return_pointwise:
                return np.inf, np.zeros_like(Mu), None
            return np.inf, np.zeros_like(Mu)

        # Compute log-pdf
        log_pdf = np.empty_like(Mu)
        small_xi_mask = np.abs(Xi) < 1e-6

        # Gumbel limit (ξ ≈ 0)
        if np.any(small_xi_mask):
            z_g = z[small_xi_mask]
            s_g = Sgm[small_xi_mask]
            log_pdf[small_xi_mask] = -np.log(s_g) - z_g - np.exp(-z_g)

        # General case (ξ ≠ 0)
        if np.any(~small_xi_mask):
            z_n = z[~small_xi_mask]
            s_n = Sgm[~small_xi_mask]
            x_n = Xi[~small_xi_mask]
            log_pdf[~small_xi_mask] = (
                -np.log(s_n)
                - (1 / x_n + 1) * np.log1p(x_n * z_n)
                - (1 + x_n * z_n) ** (-1 / x_n)
            )

        # Negative log-likelihood
        tNll = -np.sum(log_pdf, axis=0, keepdims=True)
        total_nll = np.sum(tNll)

        if return_pointwise:
            return total_nll, tNll, log_pdf
        else:
            return total_nll, tNll
        
    def posterior_predictive_nll(self, test_data, samples, time_steps=None):
        """
        Compute posterior predictive negative log-likelihood and log-predictive density.

        Returns:
        nll_pred : float  # = - log p(y_test | posterior)
        log_pred : float  # =  log p(y_test | posterior)
        """
        if time_steps is None:
            time_steps = np.arange(len(test_data))

        if isinstance(samples, np.ndarray) and samples.ndim == 2:
            sample_list = [np.ravel(samples[i, :]) for i in range(samples.shape[0])]
        else:
            raise TypeError("Samples must be a 2D numpy array")
        sample_list = list(samples)

        n_total = len(sample_list)

        loglik_values = np.empty(n_total, dtype=float)

        for i, params in enumerate(sample_list):
            params_vec = params

            total_nll, _ = self.log_likelihood(params_vec, time_steps, data=test_data)
            loglik_values[i] = -total_nll  # convert NLL -> log-likelihood
        
        # If we get -inf values, then replace with -100
        loglik_values[np.isneginf(loglik_values)] = -1000

        log_pred = logsumexp(loglik_values) - np.log(len(loglik_values))
        nll_pred = -log_pred

        return nll_pred, log_pred, loglik_values


    def log_prior(self, params, time_steps):
        """
        Vectorized prior constraints for GEV parameters.
        Much faster than the looped version.
        """
        # Check for sigmoid-type model constraint
        if self.param_structure[3] == 1:
            # Just check c parameters directly
            c_vec = np.array([params[2], params[4], params[6]])
            if np.any((c_vec <= -0.5) | (c_vec >= 10)):
                return -np.inf

        # Build parameter arrays (nT x 3)
        Mu, Sgm, Xi = self.build_parameter_arrays(params, time_steps)
        data_np = self.data.values  # (nT x 3)

        t0 = 1 + Xi * (data_np - Mu) / Sgm

        if self.upper_limit is not None:
            # This is used only for the cross-validation to avoid out-of-sample observations causing issues
            invalid_mask = (
                (Sgm <= 0) |        # invalid scale
                (Xi <= -1) |        # too negative shape
                (Xi > 0.2) |        # too large shape
                (t0 <= 0)  |        # domain constraint violated
                (Xi <= 0) & ((Mu - Sgm/Xi) < self.upper_limit)
            )
        else:
            invalid_mask = (
                (Sgm <= 0) |        # invalid scale
                (Xi <= -1) |        # too negative shape
                (Xi > 0.2) |        # too large shape
                (t0 <= 0)           # domain constraint violated
            )

        # If any invalid parameter across any site, reject
        if np.any(invalid_mask):
            return -np.inf

        # Otherwise valid prior
        return 0
    
    def propose(
        self, params, iteration, beta, total_accepted, burn_in
    ):
        # Make sure the total_accepted array has float values
        accepted_chain = np.array(total_accepted.iloc[max(0, len(total_accepted)-1000):, :], dtype=float)


        if iteration <= burn_in:
            new_parameters = [param + np.random.normal(0, 1) * 0.1 for param in params]

        else:
            SH = sqrtm(np.cov(accepted_chain, rowvar=False))
            SH = np.real(np.array(SH))
            z1 = np.random.normal(0, 1, size=self.nPrm)
            z2 = np.random.normal(0, 1, size=self.nPrm)
            y1 = (2.38 / np.sqrt(self.nPrm)) * np.matmul(SH, z1)
            y2 = (0.1 / np.sqrt(self.nPrm)) * z2
            new_parameters = params + (1 - beta) * y1 + beta * y2

        return new_parameters
    
    def acceptance_prob(self, old_params, new_params, time_steps):
        log_prior_old = self.log_prior(old_params, time_steps)
        log_prior_new = self.log_prior(new_params, time_steps)
        log_likelihood_old, tNll_old = self.log_likelihood(old_params, time_steps)
        log_likelihood_new, tNll_new = self.log_likelihood(new_params, time_steps)

        log_ratio = (-1*log_likelihood_new + log_prior_new) - (
            -1*log_likelihood_old + log_prior_old
        )

        return np.exp(log_ratio), log_likelihood_old, tNll_old
    
    def metropolis_hastings(
        self, n_samples, n2plt, burn_in=1000, thinning=10, beta=0.5, NGTSTR=0.1
    ):
        samples, total_accepted = self.setup_dataframes()
        ar = pd.DataFrame(columns=["acceptance_rate"])
        nloglikelihood = pd.DataFrame(columns=["negative_log_likelihood"])
        t_nloglikelihood = pd.DataFrame(columns=[12, 24, 58])

        current_params = self.initial_params
        time_steps = np.linspace(0, 1, len(self.data.iloc[:, 0]))

        for i in range(n_samples):
            if i <= burn_in:
                for j in range(self.nPrm):
                    new_params = current_params.copy()

                    new_params[j] = new_params[j] + np.random.normal(0, 1) * NGTSTR

                    acceptance_probability, nll, tNll = self.acceptance_prob(
                        current_params, new_params, time_steps
                    )

                    if np.random.uniform(0, 1) < acceptance_probability:

                        current_params = new_params.copy()

                nloglikelihood = pd.concat(
                    [nloglikelihood, pd.DataFrame({"negative_log_likelihood": [nll]})],
                    ignore_index=True,
                )

                t_nloglikelihood = pd.concat(
                    [
                        t_nloglikelihood,
                        pd.DataFrame(
                            tNll,
                            columns=[12, 24, 58]
                        ),
                    ],
                    ignore_index=True,
                )

                total_accepted = pd.concat(
                    [
                        total_accepted,
                        pd.DataFrame(
                            [current_params], columns=total_accepted.columns
                        ),
                    ],
                    ignore_index=True,
                )

            else:
                
                new_params = self.propose(
                    current_params,
                    i,
                    beta,
                    total_accepted,
                    burn_in,
                )
                
                acceptance_probability, nll, tNll = self.acceptance_prob(
                    current_params, new_params, time_steps
                )

                if np.random.uniform(0, 1) < acceptance_probability:

                    current_params = new_params.copy()
                
                nloglikelihood = pd.concat(
                    [nloglikelihood, pd.DataFrame({"negative_log_likelihood": [nll]})],
                    ignore_index=True,
                )

                t_nloglikelihood = pd.concat(
                    [
                        t_nloglikelihood,
                        pd.DataFrame(
                            [tNll.flatten()], columns=[12, 24, 58]  # Assigning correct labels
                        ),
                    ],
                    ignore_index=True,
                )

                total_accepted = pd.concat(
                    [
                        total_accepted,
                        pd.DataFrame(
                            [current_params], columns=total_accepted.columns
                        ),
                    ],
                    ignore_index=True,
                )

            if i >= n2plt and i % thinning == 0:
                samples = pd.concat(
                    [samples, pd.DataFrame([current_params], columns=samples.columns)],
                    ignore_index=True,
                )

            acceptance_rate = len(total_accepted) / (i + 1)

            ar = pd.concat(
                [ar, pd.DataFrame([acceptance_rate], columns=["acceptance_rate"])],
                ignore_index=True,
            )

        return samples, total_accepted, ar, nloglikelihood, t_nloglikelihood

    def find_starting_parameters(self):

        """
        This function aims to find suitable starting parameters pre mcmc.

        INPUT:
        OUTPUT:
        """

        # Here we simply fit a gumble distribution, setting xi = 0
        combined_data = self.data[['12', '24', '58']].values.ravel()

        beta = (np.sqrt(6) * np.std(combined_data)) / np.pi
        mu = np.mean(combined_data)
        xi = 0.2

        PrmCns=np.zeros(self.nPrm)

        param_idx = 0

        # Loop over the parameter structure (constant, linear, quadratic terms)

        for param_type, default_value in zip(self.param_structure, [mu, beta, xi]):
            # Assign the constant term (degree 0)
            PrmCns[param_idx] = default_value
            param_idx += 1

            # Assign zeros for linear and quadratic terms across 3 scenarios
            for degree in range(1, param_type + 1):  # Iterate over linear and quadratic terms
                for _ in range(3):  # Three scenarios for each degree
                    PrmCns[param_idx] = 0  # Initialise higher-order terms to zero
                    param_idx += 1

        return PrmCns

    def percent_point(self, q, c, mu, scale):
        """
        Compute the percent-point function (PPF) for the generalized extreme value (GEV) distribution.
        
        Parameters:
        q : float or array-like
            Probability (quantile) in range (0,1).
        c : float
            Shape parameter.
        mu : float, optional
            Location parameter (default=0).
        sigma : float, optional
            Scale parameter (default=1), must be positive.
        
        Returns:
        x : float or array-like
            The inverse CDF value corresponding to q.
        """

        q, c, mu, scale = map(np.asarray, (q, c, mu, scale))

        if np.any(scale) <= 0:
            raise ValueError("Scale parameter scale must be positive.")
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("q must be in the open interval (0,1).")
        
        if np.any(c) == 0:
            return mu - scale * np.log(-np.log(q))
        else:
            return mu + (scale / c) * ((-np.log(q)) ** (-c) - 1)
    
    def plot_return_values(self, samples):

        RtrPrd = 100
        n_samples = len(samples)

        t_start = (2025 - 2015) / (2100 - 2015)
        t_end = (2125 - 2015) / (2100 - 2015)

        RV_Start = np.zeros((n_samples, 3))
        RV_End = np.zeros((n_samples, 3))
        RV_Delta = np.zeros((n_samples, 3))

        start_params = {
            "Mu": np.zeros((n_samples, 3)),
            "Sgm": np.zeros((n_samples, 3)),
            "Xi": np.zeros((n_samples, 3)),
        }

        end_params = {
            "Mu": np.zeros((n_samples, 3)),
            "Sgm": np.zeros((n_samples, 3)),
            "Xi": np.zeros((n_samples, 3)),
        }

        indices = {
            "Mu": 0,
            "Sgm": 3 * self.param_structure[0] + 1,
            "Xi": 3 * (self.param_structure[0] + self.param_structure[1]) + 2,
        }

        for iS in range(3):

            for param in ["Mu", "Sgm", "Xi"]:

                param_idx = ["Mu", "Sgm", "Xi"].index(param)
                deg = self.param_structure[param_idx]

                terms = {
                    "C": samples.iloc[:, indices[param]],  # Constant term
                    "L": samples.iloc[:, indices[param] + deg*iS + 1] if self.param_structure[["Mu", "Sgm", "Xi"].index(param)] >= 1 else 0, # Linear term
                    "Q": samples.iloc[:, indices[param] + deg*iS + 2] if self.param_structure[["Mu", "Sgm", "Xi"].index(param)] >= 2 else 0, # Quadratic term
                }
                
                if (self.param_structure[3] == 1) & (param == "Mu"):
                    a = terms["C"]
                    b = terms["L"]
                    c = terms["Q"]

                    start_params[param][:, iS] = self.sigmoid_mu(t_start, a, b, c)
                    end_params[param][:, iS] = self.sigmoid_mu(t_end, a, b, c)
                
                else:

                    start_params[param][:, iS] = terms["C"] + terms["L"] * t_start + terms["Q"] * t_start**2
                    end_params[param][:, iS] = terms["C"] + terms["L"] * t_end + terms["Q"] * t_end**2
            
            RV_Start[:, iS] = self.percent_point(
                1 - 1 / RtrPrd, 
                c=start_params["Xi"][:, iS], 
                mu=start_params["Mu"][:, iS], 
                scale=start_params["Sgm"][:, iS]
            )

            RV_End[:, iS] = self.percent_point(
                1 - 1 / RtrPrd, 
                c=end_params["Xi"][:, iS], 
                mu=end_params["Mu"][:, iS], 
                scale=end_params["Sgm"][:, iS]
            )

        RV_Delta = RV_End - RV_Start
        RV_Delta = pd.DataFrame(RV_Delta, columns=["12", "24", "58"])

        return RV_Delta
    

    def run(self, n_samples, n2plt, burn_in=1000, thinning=1, beta=0.05, NGTSTR=0.1):
        """
        Note that the genextremefunction uses the convention for the sign of the shape
        given in the documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
        We compensate for this when displaying graphs, but keep it for computation purposes.
        """

        samples, total_accepted, ar, nloglikelihood, t_nloglikelihood = self.metropolis_hastings(
            n_samples, n2plt, burn_in, thinning, beta
        )

        return samples, total_accepted, ar, nloglikelihood, t_nloglikelihood


        




