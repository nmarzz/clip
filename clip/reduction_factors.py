import numpy as np
import scipy
from scipy.special import erfc

from scipy import stats
from scipy.stats import norm as scipy_normal
import scipy
import numpy as np

tpdf = stats.t.pdf

def F(z):
    return scipy.special.erf(z/np.sqrt(2)) - np.sqrt(2 / np.pi) * z * np.exp(-z**2 / 2)

def mu_GAU(risk,eta,c):
    # Gaussian data gaussian noise with var eta**2
    risk = risk + eta**2/2
    return F(c / np.sqrt(2 * risk)) + c* np.exp(-c**2 / 4 / risk) / np.sqrt(3.1415 * risk)

def nu_GAU(risk,eta,c):
    # Gaussian data gaussian noise with var eta**2            
    risk = risk + eta**2/2

    return (2 * risk * F(c / np.sqrt(2 * risk)) + c**2 * erfc(c / np.sqrt(4 * risk)))

def mu_STU(risk,scale,df,c):
    # compute P(not clipping) for Gaussian data and student-t (with df degrees freedom) noise
    range = np.linspace(-c,c,200)
    pdf_vals = sum_pdf_t(range,risk,scale,df)

    return scipy.integrate.simpson(pdf_vals, x = range)

def nu_STU(risk,scale,df,c):

    range = np.linspace(-c,c,200)
    pdf_vals = sum_pdf_t(range,risk,scale,df)

    integrand = range**2 * pdf_vals
    truncated_var = scipy.integrate.simpson(integrand, x = range)


    # compute P(clipping) = 1 - P(not clipping) = 1 - mu_STU
    p_clip = 1 - mu_STU(risk,scale,df,c)

    return (truncated_var + c**2 * p_clip)

def sum_pdf_t(x_values, risk, sigma, df):
    v = np.sqrt(2 * risk)
    v_sqrt_2pi = v * np.sqrt(2 * np.pi)

    # Precompute gau_pdf for the t_range
    t_range = np.linspace(-30 * sigma, 30 * sigma, 100)
    gau_values = np.exp(-0.5 * (t_range / v)**2) / v_sqrt_2pi

    # Prepare x_values as a 2D array for broadcasting
    x_values = np.atleast_2d(x_values).T

    # Compute t_pdf for each x_values - t_range in a vectorized manner
    t_pdf_values = tpdf(x_values - t_range, df, scale=sigma)

    # Calculate the integrand for all x_values
    integrand_values = gau_values * t_pdf_values

    # Integrate over t_range using Simpson's rule for all x_values
    results = scipy.integrate.simpson(integrand_values, x=t_range, axis=1)

    return results

def mu_GMM(risk, weights, means, covariances, c):
    # Compute P(not clipping) for Gaussian mixture noise
    x_range = np.linspace(-c, c, 200)
    pdf_vals = gmm_pdf(x_range, risk, weights, means, covariances)

    return scipy.integrate.simpson(pdf_vals, x=x_range)

def nu_GMM(risk, weights, means, covariances, c):
    x_range = np.linspace(-c, c, 200)
    pdf_vals = gmm_pdf(x_range, risk, weights, means, covariances)

    integrand = x_range**2 * pdf_vals
    truncated_var = scipy.integrate.simpson(integrand, x=x_range)

    # Compute P(clipping) = 1 - P(not clipping) = 1 - mu_GMM
    p_clip = 1 - mu_GMM(risk, weights, means, covariances, c)

    return (truncated_var + c**2 * p_clip)

def gmm_pdf(x_values, risk, weights, means, covariances):
    """
    Computes the probability density function of a Gaussian Mixture Model (GMM) noise and gaussian data
    at given x_values using the provided weights, means, and covariances.
    """

    v = np.sqrt(2 * risk)
    v_sqrt_2pi = v * np.sqrt(2 * np.pi)

    # Precompute gau_pdf for the t_range
    max_sigma = np.max(covariances)
    t_range = np.linspace(-30 * max_sigma, 30 * max_sigma, 200)
    gau_values = np.exp(-0.5 * (t_range / v)**2) / v_sqrt_2pi

    # Prepare x_values as a 2D array for broadcasting
    x_values = np.atleast_2d(x_values).T

    # Compute GMM PDF for each x_values - t_range
    gmm_values = np.zeros((x_values.shape[0], t_range.shape[0]))
    for weight, mean, cov in zip(weights, means, covariances):
        # GMM component PDFs
        gmm_values += weight * scipy_normal.pdf(x_values - t_range, loc=mean[0], scale=np.sqrt(cov[0][0]))
    
    # Calculate the integrand for all x_values
    integrand_values = gau_values * gmm_values

    # Integrate over t_range using Simpson's rule for all x_values
    results = scipy.integrate.simpson(integrand_values, x=t_range, axis=1)

    return results
