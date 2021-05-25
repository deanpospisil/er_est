#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:26:15 2021.

Functions for estimating r^2 between neural tuning curve and model (n2m)
and between two neural tuning curves (n2n). See examples ipynb's to see example
code applying estimators to simulated data and plotting results.

For n2m functions see:
Pospisil and Bair (2020) The unbiased estimation of the fraction of variance
explained by a model. BioArxiv.

For n2n functions see:
Pospisil and Bair (2021) The unbiased estimation of $r^2$ between two sets of
noisy neural responses. BioArxiv.

For the split version of n2n see:
Pospisil and Bair (2021) Accounting for biases in the estimation of
neuronal signal correlation. Journal of Neuroscience.

@author: dean
"""

import numpy as np
from scipy import stats

##################### n2m ###############################################
def r2er_n2m(x, y):
    """Neuron to model approx. unbiased estimator of r2er.

    Parameters
    ----------
    x : numpy.ndarray
        m observations model predictions
    y : numpy.ndarray
        N neurons X n repeats X m observations of data

    Returns
    -------
    r2er : an estimate of the r2 between model and expected value of data
    --------
    """
    n, m = np.shape(y)[-2:]
    # estimate of trial-to-trial variability
    sig2_hat = np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True)
    # mean center the model
    x_ms = x - np.mean(x, keepdims=True)
    # get average across repeats
    y = np.mean(y, -2, keepdims=True)
    # mean center data average
    y_ms = y - np.mean(y, (-1), keepdims=True)

    # dot product of mean centered model and data
    # are biased numerator
    xy2 = np.sum((x_ms*y_ms), -1, keepdims=True)**2

    # individual variances of model and data
    x2 = np.sum(x_ms**2, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    # biased denominator
    x2y2 = x2*y2
    r2 = xy2/x2y2
    # unbias numerator and denominator
    ub_xy2 = xy2 - sig2_hat/n * x2
    ub_x2y2 = x2y2 - (m-1)*sig2_hat/n*x2

    # form ratio of individually unbiased estimates from r^2_er
    hat_r2er = ub_xy2/ub_x2y2

    return hat_r2er, r2


def sim_n2m(r2er, sig2, d2y, n, m, n_exps, verbose=True):
    """Simulate neuron with arbitrary correlation to model.

    Parameters
    ----------
    r2er : numpy.double
         pearsons r^2 between tuning curve and model
    sig2 : numpy.double
          trial-to-trial variability
    d2y : numpy.double
          dynamic range of neuron
    m : numpy.double
        number of stimuli/observations
    n : numpy.double
        number of repeats
    n_exps : numpy.double
             number of IID simulations with parameters above
    verbose: boolean
            if true when ensuring all parameters within domain will
            print message if a parameter was changed.

    Returns
    -------
    x : m long vector model predictions with r2 to expected responses of neuron
    y : n_exps X n repeats X m stimuli simulations from parameters
    --------
    """
    # make sure all parameters are within domain of sim
    n = int(n)
    m = int(m)
    if r2er < 0:
        if verbose:
            print('r2 below 0, setting to 0')
        r2er = 0
    elif r2er > 1:
        if verbose:
            print('r2 above 1, setting to 1')
        r2er = 1
    if d2y < 0:
        if verbose:
            print('d2y below 0, setting to 0')
        d2y = 0
    # calculate angle between model and neuron expected response vector
    angle = np.arccos(r2er**0.5)
    s = np.linspace(0, 2*np.pi*(1-1/m), m)
    # model is cosine
    x = np.cos(s)
    # neural responses are shifted cosine by appropriate angle
    yu = np.cos(angle + s)
    # scale neural means to desired dynamic range
    yu = (d2y**0.5)*(yu/(yu**2).sum()**0.5)
    # now draw sample from normal IID across n_exps
    y = np.random.normal(loc=yu, scale=sig2**0.5,
                         size=(n_exps,) + (n,) + yu.shape)
    return x, y
# %% n2m CI


def sample_post_s2_d2(y, n_samps):
    """Draw posterior distribution of sig2 and d2.

    draw n_samps samples from posterior of sig2, d2 (trial-to-trial variability
        and dynamic range) given hat_s2, hat_d2 (estimates). Uses
        metropolis-hastings algorithm to get draws.

    Parameters
    ----------
    y : numpy.ndarray
        n repeats X m observations array of data
    n_samps: int
            how many simulations with these parameters to draw

    Returns
    -------
    trace : numpy.ndarray
        2 X n_samps drawn from posterior (d^2, sigma^2)
    p:  the proportion of times a candidate parameter was accepted, a
        metropolis hastings metric
    --------
    """
    n, m = y.shape
    # get the observed statistics
    hat_sig2 = y.var(0, ddof=1).mean(0)  # estimated trial-to-trial var
    hat_d2 = y.mean(0).var(0, ddof=1)  # estimated dynamic range

    # initialize sampling of parameters at estimates of parameters
    sig2_curr = hat_sig2
    d2_curr = hat_d2

    # scaled non-central chi squared pdf of dynamic range estimate.
    df = m - 1
    scale = sig2_curr/(n * (m - 1))
    nc = (d2_curr * m)/(sig2_curr / n)
    vd2 = stats.ncx2.var(df, nc, scale=scale)
    # scaled central chi squared pdf of sample variance estimate.
    df = m*(n - 1)
    nc = 0
    scale = sig2_curr/(m * (n - 1))
    vs2 = stats.ncx2.var(df, nc, scale=scale)

    accept = np.zeros(n_samps)
    trace = np.zeros((2, n_samps))
    u = np.random.uniform(0, 1, n_samps)
    for i in (range(n_samps)):
        # proposal distribution is mv normal
        sig2_cand, d2_cand = np.random.normal(loc=[(sig2_curr), (d2_curr)],
                                              scale=[vs2**0.5, vd2**0.5])

        # if candidate likelihood will be 0 skip calc and do dont accept
        if not (sig2_cand <= 0 or d2_cand <= 0):

            # calculate likelihoods of candiates and current parameters
            scale = sig2_cand/(n*(m-1))
            fd2_cand = (stats.ncx2._pdf(hat_d2*(scale**-1),
                                        df=m-1,
                                        nc=(d2_cand*m)/(sig2_cand/n)
                                        )*(scale**-1))
            scale = sig2_curr/(n*(m-1))
            fd2_curr = stats.ncx2._pdf(hat_d2*(scale**-1),
                                       df=m-1,
                                       nc=(d2_curr*m)/(sig2_curr/n)
                                       )*(scale**-1)

            scale = sig2_cand/(m*(n-1))
            fs2_cand = stats.ncx2._pdf(hat_sig2*(scale**-1),
                                       df=m*(n-1),
                                       nc=0)*(scale**-1)
            scale = sig2_curr/(m*(n-1))
            fs2_curr = stats.ncx2._pdf(hat_sig2*(scale**-1),
                                       df=m*(n-1),
                                       nc=0)*(scale**-1)

            # take ratio of two current and candidate likelihoods
            a = (fs2_cand*fd2_cand)/(fs2_curr*fd2_curr)
            if a >= 1:
                # current becomes candidate
                sig2_curr = sig2_cand
                d2_curr = d2_cand
                accept[i] = 1
            else:
                if u[i] < a:
                    # current becomes candidate
                    accept[i] = 1
                    sig2_curr = sig2_cand
                    d2_curr = d2_cand
        # add current to trace.
        trace[:, i] = [d2_curr, sig2_curr]
    accept_p = np.mean(accept)
    return trace, accept_p


def gr_rhat(trace):
    """Calculate gelman rubin statistic rhat.

    Should be less than 1.2 for a MCMC chain that has converged.  Trace should
    be n samples by m chains for single variable.
    """
    n, m = trace.shape  # n samples, m chains
    theta_m = trace.mean(0)
    sig2_m = trace.var(0, ddof=1)

    B = n*np.var(theta_m, ddof=1)
    W = np.mean(sig2_m)

    hat_V = ((n - 1)/n)*W + ((m+1)/(m*n))*B
    hat_R = hat_V/W

    return hat_R


def get_emp_dist_r2er_n2m(r2er_check, trace, m, n, n_r2er_sims=100):
    """Sample from hat_r^2_{ER} with draws from posterior sig2, d2 trace."""
    sig2_post = trace[1]  # trial-to-trial variance
    d2m_post = trace[0]*m  # dynamic range

    # randomly sample from post-dist
    sample_inds = np.random.choice(len(sig2_post),
                                   size=n_r2er_sims,
                                   replace=True)
    res = np.zeros(n_r2er_sims)
    for j in range(n_r2er_sims):
        k = sample_inds[j]  # get random indices for posterior samples
        x, y = sim_n2m(r2er=r2er_check,
                       sig2=sig2_post[k],
                       d2y=d2m_post[k],
                       n=n,
                       m=m, n_exps=1)
        res[j] = (r2er_n2m(x.squeeze(), y)[0]).squeeze()  # calculate hat_r2er
    return res


def find_sgn_p_cand_n2m(r2er_cand, r2er_hat_obs, alpha_targ, trace, m, n,
                        p_thresh=0.01, n_r2er_sims=1000):
    """Find if value of cdf of r2er_cand at r2er_hat_obs is >, <, or == alpha.

    helper function for find_r2er_w_quantile_n2m
    """
    z_thresh = stats.norm.ppf(1.-p_thresh)
    res = get_emp_dist_r2er_n2m(r2er_cand, trace, m, n,
                                n_r2er_sims=n_r2er_sims)

    count = (res < r2er_hat_obs).sum()

    # z-test
    z = ((count - alpha_targ*n_r2er_sims) /
         (n_r2er_sims*alpha_targ*(1-alpha_targ))**0.5)

    if z > z_thresh:  # signif. above desired count
        sgn_p_cand = 1
    elif -z > z_thresh:  # signif. below desired count
        sgn_p_cand = -1
    else:  # NS different
        sgn_p_cand = 0
    return sgn_p_cand, res


def find_r2er_w_quantile_n2m(r2er_hat_obs, alpha_targ, trace, m, n, n_splits=6,
                             p_thresh=1e-2, n_r2er_sims=100, int_l=0, int_h=1):
    """Find r2er with r2er_hat distribution  CDF(r2er_hat_obs)=alpha_targ.

    Helper function for ecci_r2er_n2m.
    """
    # we first check at the highest and lowest possible values
    sgn_p_cand_h, res = find_sgn_p_cand_n2m(r2er_cand=int_h,
                                            r2er_hat_obs=r2er_hat_obs,
                                            alpha_targ=alpha_targ,
                                            trace=trace, m=m, n=n,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims)

    sgn_p_cand_l, res = find_sgn_p_cand_n2m(r2er_cand=int_l,
                                            r2er_hat_obs=r2er_hat_obs,
                                            alpha_targ=alpha_targ,
                                            trace=trace, m=m, n=n,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims)

    # if the highest possible value is too low, or is NS different from
    # desired alpha return it as the r2_er with the desired alpha.
    if sgn_p_cand_h == 1 or sgn_p_cand_h == 0:
        return int_h, res

    # if the lowest possible value is too high, or is NS different from
    # desired alpha return it as the r2_er with the desired alpha
    if sgn_p_cand_l == -1 or sgn_p_cand_l == 0:
        return int_l, res

    # if endpoints are not r2_er desired then begin iterative bracketing
    # with max of n_splits iterations.
    for split in range(n_splits):
        # choose random point in current r2_Er range
        c_cand = np.random.uniform(int_l, int_h)

        # evaluate it
        sgn_p_cand, res = find_sgn_p_cand_n2m(r2er_cand=c_cand,
                                              r2er_hat_obs=r2er_hat_obs,
                                              alpha_targ=alpha_targ,
                                              trace=trace, m=m, n=n,
                                              p_thresh=p_thresh,
                                              n_r2er_sims=n_r2er_sims)
        # if it was too high then set it as the  new upper end of the interval
        if sgn_p_cand == -1:
            int_h = c_cand
        # if it was too low then set it as the new lower end
        elif sgn_p_cand == 1:
            int_l = c_cand
        # if it was NS diffent then return the candidate
        else:
            return c_cand, res

    return c_cand, res


def ecci_r2er_n2m(x, y, alpha_targ=0.1, n_r2er_sims=1000,  p_thresh=0.01,
                  n_splits=20, trace=None):
    """Algorithm to find alpha_targ level confidence intervals for hat_r2_{ER}.

    Parameters
    ----------
    x : numpy.ndarray
        m observations model predictions
    y : numpy.ndarray
        n repeats X m observations array of data
    alpha_targ : float
        desired alpha level (0,1) (proportion of CI's containing r^2_er)
    n_r2er_sims : int
        how many samples of n_r2er_sims hat_r2_{ER} to draw to calculate
        quantile estimates and how long trace should be
    p_thresh: float
        p-value below which we will reject null hypothesis that a candidate
        r2_ER gives alpha_targ
     n_splits : int
         for iterative bracketing algorithm find_cdf_pos the number of times it
         will split so 6 times gives a range of potential value ~2**-6 = 0.015
         in length.
    trace : numpy.ndarray
        if trace for posterior has already been calculated you can use it here
        if none then finds posterior.

    Returns
    -------
    ll : float
        lower end of confidence interval
    ul : float
        upper end of confidence interval
    r2er_hat_obs : float
        estimate hat_r2_{er}

    trace : numpy.ndarray
        posterior trace

    ll_alpha : numpy.ndarray
        the distribution of hat_r^2_er assosciated with ll

    ul_alpha : numpy.ndarray
        samples from hat_r^2_er assosciated with ul
    --------
    """
    # get confidence intervals
    n, m = y.shape
    r2er_hat_obs = r2er_n2m(x.squeeze(), y)[0]
    # calculate m_traces different traces so that converge
    # stat gelmin rubin rhat can be calculated
    if trace is None:
        m_traces = 5
        traces = np.zeros((m_traces, 2, n_r2er_sims))
        for a_m in range(m_traces):

            # get posterior dist of params
            trace, p = sample_post_s2_d2(y, n_samps=n_r2er_sims)
            traces[a_m] = trace
        for i in range(2):
            rhat = gr_rhat(traces[:, i].T)
            if rhat > 1.2:
                print(('warning r_hat=' + str(np.round(rhat, 2)) +
                       ' is greater than 1.2, try increasing n_r2er_sims'))

    ul, ul_alpha = find_r2er_w_quantile_n2m(r2er_hat_obs, alpha_targ/2,
                                            trace, m, n,
                                            n_splits=n_splits,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims,
                                            int_l=0, int_h=1)
    ll, ll_alpha = find_r2er_w_quantile_n2m(r2er_hat_obs, 1-alpha_targ/2,
                                            trace, m, n, n_splits=n_splits,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims,
                                            int_l=0, int_h=1)

    return ll, ul, r2er_hat_obs, trace, ll_alpha, ul_alpha


# %% other n2m CI implementations
def percentile(x, q):
    """Percentile for bootstrap functions below."""
    inds = np.argsort(x)
    ind = int((len(x)-1)*q/100.)
    val = x[inds[ind]]
    return val


def get_npbs_ci(x, y, alpha_targ, n_bs_samples=1000):
    """Non-paramateric bootstrap of n2m r2er."""
    n, m = y.shape
    y_bs = []
    for k in range(n_bs_samples):
        _ = np.array([np.random.choice(y_obs, size=n) for y_obs in y.T]).T
        y_bs.append(_)
    y_bs = np.array(y_bs)
    r2er_bs = r2er_n2m(x.squeeze(), y_bs)[0].squeeze()

    ci = np.quantile(r2er_bs, [alpha_targ/2, 1 - alpha_targ/2]).T
    return ci


def get_pbs_ci(x, y, alpha_targ, n_pbs_samples=1000):
    """Paramateric bootstrap (MV norm) of n2m r2er."""
    n, m = y.shape
    r2er_hat_obs = r2er_n2m(x.squeeze(), y)[0].squeeze()
    hat_sig2 = np.var(y, 0).mean()
    hat_d2y = np.sum((y.mean(0)-y.mean())**2, keepdims=True) - (m-1)*hat_sig2/n
    x_new, y_new = sim_n2m(r2er=r2er_hat_obs,
                           sig2=hat_sig2,
                           d2y=hat_d2y,
                           n=n,
                           m=m,
                           n_exps=n_pbs_samples)

    r2er_pbs = r2er_n2m(x_new.squeeze(), y_new)[0].squeeze()
    ci = np.quantile(r2er_pbs, [alpha_targ/2, 1 - alpha_targ/2]).T
    return ci


def get_pbs_bca_ci(x, y, alpha_targ, n_bs_samples):
    """Paramateric bootstrap (MV norm) corrected and accel of r2er n2m."""
    z_alpha = stats.norm.ppf(alpha_targ/2.)
    z_1_alpha = stats.norm.ppf(1-alpha_targ/2.)
    n, m = y.shape
    r2er_hat_obs = r2er_n2m(x.squeeze(), y)[0].squeeze()

    hat_sig2 = np.var(y, 0).mean()
    hat_d2y = np.sum((y.mean(0)-y.mean())**2, keepdims=True) - (m-1)*hat_sig2/n
    x_new, y_new = sim_n2m(r2er=r2er_hat_obs,
                           sig2=hat_sig2,
                           d2y=hat_d2y,
                           n=n,
                           m=m,
                           n_exps=n_bs_samples)

    r2er_pbs = r2er_n2m(x_new.squeeze(), y_new)[0].squeeze()

    # jack knife
    jack_r2er = []
    for i in range(m):
        jack_y = np.array([y[:, j] for j in range(m) if i != j]).T
        jack_x = np.array([x[j] for j in range(m) if i != j])
        _ = r2er_n2m(jack_x, jack_y)[0].squeeze()
        jack_r2er.append(_.squeeze())

    jack_r2er = np.array(jack_r2er)
    jack_r2er_dot = jack_r2er.mean()

    # bias correction factor
    bias_factor = np.mean(r2er_pbs < r2er_hat_obs)
    z_hat_0 = stats.norm.ppf(bias_factor)

    # acceleration accounts for parameter-variance relationship as linear
    a_hat = (np.sum((jack_r2er_dot - jack_r2er)**3) /
             (6.*(np.sum((jack_r2er_dot - jack_r2er)**2))**(3/2.)))

    # in case estimate is above or below all BS samples.
    if z_hat_0 == np.inf:
        alpha_1 = 1
        alpha_2 = 1
    elif z_hat_0 == -np.inf:
        alpha_1 = 0
        alpha_2 = 0
    else:
        alpha_1 = stats.norm.cdf(z_hat_0 +
                                 (z_alpha + z_hat_0) /
                                 (1 - a_hat*(z_alpha + z_hat_0)))
        alpha_2 = stats.norm.cdf(z_hat_0 +
                                 (z_1_alpha + z_hat_0) /
                                 (1 - a_hat*(z_1_alpha + z_hat_0)))

    ci_low = percentile(r2er_pbs, alpha_1*100.)
    ci_high = percentile(r2er_pbs, alpha_2*100.)

    return [ci_low, ci_high]


# %% other n2m estimators implementation
def rand_splits(n, k):
    """Give indices for splitting into non-overlapping trials.

    Parameters
    ----------
    n : int
        Number of trials to split across.
    k : int
        How many random splits to make.

    Returns
    -------
    c : np.array
        Indices of non-overlapping random splits of trials.

    """
    c = []
    for i in range(k):
        ab = np.random.permutation(n)
        a = ab[:int(n/2)]
        b = ab[int(n/2):]
        c.append([a, b])
    return np.array(c)


def r2_SE_corrected(model, data):
    """Estimator Pasupathy A, Connor CE (2001) Shape Representation ...

    From section Response Measurement error and personal communication
    with Dr Pasupathy. model should m long and data n x m.
    """
    r2 = np.corrcoef(data.mean(0), model)[0, 1]**2
    nv = np.mean(np.var(data, 0, ddof=0))
    tot_var = np.var(data, ddof=0)
    frac_explainable_var = 1 - nv/tot_var
    r2_norm = r2/frac_explainable_var

    return r2_norm


def upsilon(model, data, d=2):
    """Estimator Haefner RalfM, Cumming BruceG (2008) An improved estimator...

    In paper d is the dimension of the model being fit
    under assumptions of derivation the model is a linear model
    d being the number of parameters. d=2 is default where
    it is assumed intercept and single term are fit
    model should m long and data n x m.
    """
    n, m = data.shape
    hat_sig2 = np.mean(np.var(data, 0, ddof=1))/n

    SS_res = np.sum((data.mean(0)-model)**2.)/(hat_sig2)
    SS_tot = np.sum((data.mean(0)-data.mean())**2.)/(hat_sig2)

    SS_res_ub = SS_res - (m*(n-1)*(m-d))/(m*(n-1)-2)
    SS_tot_ub = SS_tot - (m*(n-1)*(m-1))/(m*(n-1)-2)

    upsilon = 1 - SS_res_ub/SS_tot_ub

    return upsilon


def r2_SB_normed(model, data, n_split=10):
    """Estimator Yamins DLK, ..., DiCarlo JJ (2014) Performance-optimized ...

    Taken from yamins et al 2014 PNAS supp info page 2 para. 3 and personal
    communication Martin Schrimpf.
    """
    n, m = data.shape
    r_split = np.array([np.corrcoef(data[a].mean(0),
                                    data[b].mean(0))[0, 1]
                        for (a, b) in rand_splits(n, n_split)]).mean()
    r_split_sb = (2*r_split)/(1 + r_split)

    r = np.corrcoef(data.mean(0), model)[0, 1]

    r2_norm = (r/r_split_sb)**2.
    return r2_norm


def cc_norm_split(model, data, n_split=10):
    """Estimator Hsu A, Borst A, Theunissen FE (2004) Quantifying variabil...

    As derived in schoppe et al 2016.
    """
    n, m = data.shape
    cc_abs = np.corrcoef(model, data.mean(0))[0, 1]
    cc_half = np.array([np.corrcoef(data[a].mean(0),
                                    data[b].mean(0))[0, 1]
                        for (a, b) in rand_splits(n, n_split)]).mean()
    cc_max = np.sqrt(2/(1+np.sqrt(1/(cc_half**2.))))
    cc_norm = cc_abs/cc_max

    return cc_norm


def normalized_spe(model, data):
    """Estimator Sahani M, Linden JF (2003) How Linear are Auditory ...

    As described in Schoppe et al 2016.
    """
    n, m = data.shape
    data_m = data.mean(0)
    data_m_v = np.var(data_m)
    TP = np.mean(np.var(data, 1))
    SP = (1/(n-1))*(n*data_m_v - TP)
    SPE = data_m_v - np.var(data_m - model)
    SPE_norm = SPE/SP

    return SPE_norm


def cc_norm(model, data):
    """Estimator  Schoppe O, ..., Schnupp JWH (2016) Measuring the Perfor...

    Numerically identical to normalized_spe for 2 param linear model.
    """
    n, m = data.shape
    data_m = data.mean(0)
    data_m_v = np.var(data_m)
    TP = np.mean(np.var(data, 1))
    SP = (1/(n-1))*(n*data_m_v - TP)

    cov_model_data = np.cov(model, data_m, ddof=0)[0, 1]
    var_model = np.var(model, ddof=0)
    CC_norm = cov_model_data/np.sqrt(var_model*SP)

    return CC_norm


def cc_norm_bs(model, data, n_split=10):
    """Estimator Kindel WF, ..., Zylberberg J (2019) Using deep learning ...

    From personal communication Dr Zylberberg. They cite Schoppe but use
    parametric bootstrap normal simulation for cc_max.
    """
    n, m = data.shape
    data_m = data.mean(0)
    data_sim = np.random.normal(loc=data_m, scale=data.std(0).mean(),
                                size=(n_split, n, m))
    cc_max = np.mean([np.corrcoef(data_sim[i].mean(0), data_m)[0, 1]
                      for i in range(n_split)])
    cc_abs = np.corrcoef(data_m, model)[0, 1]

    cc_norm = cc_abs/cc_max
    return cc_norm


def feve_cadena(model, data):
    """Estimator Cadena SA, ..., Ecker AS (2019) Deep convolutional models ...

    From personal communication Cadena and github code assosciated with pub.
    """
    resp = data
    pred = model[np.newaxis]

    mse = np.mean((pred - resp)**2)  # MSE total scalar
    obs_var = np.mean((np.var(resp, axis=0, ddof=1)), axis=0)
    total_variance = np.var(resp, axis=(0, 1), ddof=1)

    explainable_var = (total_variance - obs_var)

    eve = 1 - (mse - obs_var) / explainable_var

    return eve


def r2er_n2lm(l_mod, y):
    """Neuron to linear model approx. unbiased estimator of r2er.

    Parameters
    ----------
    x : numpy.ndarray
        m observations X d variable  model predictions
    y : numpy.ndarray
        N neurons X n repeats X m observations of data

    Returns
    -------
    r2er : an estimate of the r2 between model and expected value of data
    --------
    """
    m, d = l_mod.shape
    n = y.shape[-2]
    ym = y.mean(-2)
    sig2_hat = np.mean(np.var(y, -2, ddof=1, keepdims=True),
                       -1, keepdims=True).squeeze()

    # fit linear model
    beta = np.linalg.lstsq(l_mod, ym.T, rcond=-1)[0]
    y_hat = np.dot(l_mod, beta).squeeze().T

    # unbiased estimate of residual var
    num = np.sum((ym-y_hat)**2) - (m-d)*sig2_hat/n
    # unbiased estimate total var
    den = np.sum((ym-ym.mean())**2) - (m-1)*sig2_hat/n
    # convert to explained variance
    hat_r2er = (1-num/den).squeeze()

    return hat_r2er

# %% helpful estimators


def d2_er_est(y, scale=False):
    """Approx unbiased estimator of dynamic range across expected values.

        Assumes data has equal variance across  trials and observations
        (may require variance stabilizing transform).

    Parameters
    ----------
    y : numpy.ndarray
        N neurons X n trials X m observations array

    Returns
    -------
    r2er : an estimate of the r2 between the expected values of the data
    --------
    """
    n, m = np.shape(y)[-2:]
    sig2_hat = np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True)
    ym = np.mean(y, -2, keepdims=True)
    y_ms = ym - np.mean(ym, -1, keepdims=True)
    d2 = np.sum(y_ms**2, -1, keepdims=True)

    d2_er = d2 - (m-1)*sig2_hat/n

    if scale:
        d2_er = d2_er/m

    return d2_er


def snr_er_est(y):
    """Approximately unbiased estimator of snr.

        Assumes y has equal variance across trials and observations
        (may require variance stabilizing transform).

    Parameters
    ----------
    y : numpy.ndarray
        N neurons X n trials X m observations array

    Returns
    -------
    snr_ER : an approximately unbiased estimate of snr
    --------
    """
    n, m = np.shape(y)[-2:]
    sig2_hat = np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True)
    d2_er = d2_er_est(y, scale=True)
    snr_er = d2_er/sig2_hat

    return snr_er


##################### n2n ###############################################
def r2er_n2n(x, y):
    """Neuron to neuron approx. unbiased estimator of r2er.

    Parameters
    ----------
    x : numpy.ndarray
        N neurons X n trials X m observations array
    y : numpy.ndarray
        N neurons X n trials X m observations array

    Returns
    -------
    r2er : an estimate of the r2 between the expected values of the data
    --------
    """
    n, m = np.shape(y)[-2:]

    # estimate trial-to-trial variability as average of both neurons
    sig2_hat = (np.mean(np.var(y, -2, ddof=1, keepdims=1), -1, keepdims=1) +
                np.mean(np.var(x, -2, ddof=1, keepdims=1), -1, keepdims=1))/2.

    # get trial averages of x and y and mean center
    x = np.mean(x, -2, keepdims=True)
    x_ms = x - np.mean(x, -1, keepdims=True)
    y = np.mean(y, -2, keepdims=True)
    y_ms = y - np.mean(y, -1, keepdims=True)

    # uncorrected numerator
    xy2 = np.sum((x_ms*y_ms), -1, keepdims=True)**2
    # uncorrected denominator
    x2 = np.sum(x_ms**2, -1, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    x2y2 = x2*y2

    r2 = xy2/x2y2

    # subtract off bias
    ub_xy2 = xy2 - (sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n))
    ub_x2y2 = x2y2 - (m-1)*sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n)

    r2er = ub_xy2/ub_x2y2

    return r2er, r2


def sim_n2n(r2er, sig2, d2x, d2y, n, m, n_exps, verbose=True):
    """Simulate neuron pair with arbitrary correlation to each other.

    Parameters
    ----------
    r2er : numpy.double
         pearsons r^2 between tuning curve and model
    sig2 : numpy.double
          trial-to-trial variability
    d2x : numpy.double
          dynamic range of neuron X
    d2y : numpy.double
          dynamic range of neuron Y
    n : int
        number of repeats
    m : int
        number of stimuli/observations
    n_exps : int
             number of IID simulations with parameters above
    verbose: boolean
            if true when ensuring all parameters within domain will
            print message if a parameter was changed.

    Returns
    -------
    x : n_exps X n repeats X m stimuli simulations from parameters
    y : n_exps X n repeats X m stimuli simulations from parameters
    --------
    """
    n = int(n)
    m = int(m)
    if r2er < 0:
        if verbose:
            print('r2 below 0, setting to 0')
        r2er = 0
    elif r2er > 1:
        if verbose:
            print('r2 above 1, setting to 1')
        r2er = 1
    if d2y < 0:
        if verbose:
            print('d2y below 0, setting to 0')
        d2y = 0
    if d2x < 0:
        if verbose:
            print('d2x below 0, setting to 0')
        d2x = 0

    angle = np.arccos(r2er**0.5)

    s = np.linspace(0, 2*np.pi, int(m))

    xu = np.cos(s)
    yu = np.cos(s + angle)

    xu = (d2x**0.5)*(xu/(xu**2.).sum()**0.5)
    yu = (d2y**0.5)*(yu/(yu**2.).sum()**0.5)

    x = np.random.normal(loc=xu, scale=sig2**0.5,
                         size=(n_exps,) + (n,) + xu.shape)
    y = np.random.normal(loc=yu, scale=sig2**0.5,
                         size=(n_exps,) + (n,) + yu.shape)

    return x, y

# %% alternative n2n estimator

def r2er_n2n_split(x, y, k_splits=100):
    """r2er_n2n except calculated over non-overlapping random splits.

    This should be used when trials are collected simultaneously and their may
    be noise correlation. Doesn't broadcast over multiple neuron pairs

    Parameters
    ----------
    x : numpy.ndarray
        N neurons X n trials X m observations array
    y : numpy.ndarray
        N neurons X n trials X m observations array
    k : int
        number of splits to average over

    Returns
    -------
    hat_r2er_split : an estimate of the r2 between the expected values of
                     the data averaged over k splits
    --------
    """
    n, m = y.shape[-2:]
    split_inds = np.array(rand_splits(n, k_splits))
    hat_r2er_split = np.array([r2er_n2n(x[..., a, :],
                                        y[..., b, :])[0].squeeze()
                               for (a, b) in split_inds]).mean(0)

    return hat_r2er_split


def hat_rho_0_spearman(x, y, correct_d2=True):
    """Spearman estimator of r between expected values of two rvs.

        Assumes x and y have equal variance across trials and observations
        (may require variance stabilizing transform).
        this method is attributed to
        Spearman C (1904) The proof and measurement of association between ...
        following methods of
        Adolph SC, Hardin JS (2007) Estimating Phenotypic Correlations: ...

    Parameters
    ----------
    x : numpy.ndarray
        N neurons X n trials X m observations array
    y : numpy.ndarray
        N neurons X n trials X m observations array

    Returns
    -------
    r2er : an estimate of the r2 between the expected values of the data
    --------
    """
    n, m = np.shape(y)[-2:]
    sig2_hat = (np.mean(np.var(y, -2, ddof=1, keepdims=1), -1, keepdims=1) +
                np.mean(np.var(x, -2, ddof=1, keepdims=1), -1, keepdims=1))/2.
    sig2_hat = sig2_hat/n

    x = np.mean(x, -2, keepdims=True)
    x_ms = x - np.mean(x, -1, keepdims=True)

    y = np.mean(y, -2, keepdims=True)
    y_ms = y - np.mean(y, -1, keepdims=True)

    xy = np.sum((x_ms*y_ms), -1, keepdims=True)
    x2 = np.sum(x_ms**2, -1, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    x2y2 = (x2*y2)

    if correct_d2:
        d2x = (1/(m-1))*(x2 - (m-1)*sig2_hat)  # corrected for bias
        d2y = (1/(m-1))*(y2 - (m-1)*sig2_hat)
    else:
        d2x = (1/(m-1))*x2  # Adolph and Hardin
        d2y = (1/(m-1))*y2

    A_inv = ((1+sig2_hat/d2x)*(1+sig2_hat/d2y))**0.5
    r = xy/(x2y2**0.5)

    r0 = A_inv*r

    return r0


# %% n2n ci


def sample_post_s2_d2x_d2y(y, x, n_samps):
    """Draw n_samps samples from posterior of sig2, d2x, d2y given y, x.

    Uses metropolis-hastings algorithm to get draws.

    Parameters
    ----------
    y : numpy.ndarray
        n repeats X m observations array of data
    x : numpy.ndarray
        n repeats X m observations array of data
    n_samps: int
            how many simulations with these parameters to draw

    Returns
    -------
    trace : numpy.ndarray
        2 X n_samps drawn from posterior (d^2, sigma^2)
    p:  proportion of times a candidate parameter was accepted, a metropolis
        hastings metric.
    --------
    """
    n, m = y.shape
    # get the observed statistics
    # estimated trial-to-trial var
    hat_sig2 = (y.var(0, ddof=1).mean(0) + x.var(0, ddof=1).mean(0))/2
    hat_d2y = y.mean(0).var(0, ddof=1)  # estimated dynamic range
    hat_d2x = x.mean(0).var(0, ddof=1)  # estimated dynamic range

    # initialize sampling of parameters at estimates of parameters
    sig2_curr = hat_sig2
    d2x_curr = hat_d2x
    d2y_curr = hat_d2y

    # scaled non-central chi squared pdf of dynamic range estimate.
    df = m - 1
    scale = sig2_curr/(n * (m - 1))

    ncx = (d2x_curr * m)/(sig2_curr / n)
    vd2x = stats.ncx2.var(df, ncx, scale=scale)

    ncy = (d2y_curr * m)/(sig2_curr / n)
    vd2y = stats.ncx2.var(df, ncy, scale=scale)
    # scaled central chi squared pdf of sample variance estimate.
    df = m*(n - 1)
    nc = 0
    scale = sig2_curr/(m * (n - 1))
    vs2 = stats.ncx2.var(df, nc, scale=scale)

    accept = np.zeros(n_samps)
    trace = np.zeros((3, n_samps))
    u = np.random.uniform(0, 1, n_samps)
    for i in (range(n_samps)):
        # proposal distribution is mv normal
        sig2_cand, d2x_cand, d2y_cand = np.random.normal(
            loc=[sig2_curr,
                 d2x_curr,
                 d2y_curr],
            scale=[vs2**0.5,
                   vd2x**0.5,
                   vd2y**0.5])

        # if candidate likelihood will be 0 skip calc and do dont accept
        if not (sig2_cand <= 0 or d2x_cand <= 0 or d2y_cand <= 0):

            # calculate likelihoods of candiates and current parameters
            scale = sig2_cand/(n*(m-1))
            fd2x_cand = (stats.ncx2._pdf(hat_d2x*(scale**-1),
                                         df=m-1,
                                         nc=(d2x_cand*m)/(sig2_cand/n)
                                         )*(scale**-1))
            fd2y_cand = (stats.ncx2._pdf(hat_d2y*(scale**-1),
                                         df=m-1,
                                         nc=(d2y_cand*m)/(sig2_cand/n)
                                         )*(scale**-1))

            scale = sig2_curr/(n*(m-1))
            fd2x_curr = stats.ncx2._pdf(hat_d2x*(scale**-1),
                                        df=m-1,
                                        nc=(d2x_curr*m)/(sig2_curr/n)
                                        )*(scale**-1)
            fd2y_curr = stats.ncx2._pdf(hat_d2y*(scale**-1),
                                        df=m-1,
                                        nc=(d2y_curr*m)/(sig2_curr/n)
                                        )*(scale**-1)

            scale = sig2_cand/(m*(n-1))
            fs2_cand = stats.ncx2._pdf(hat_sig2*(scale**-1),
                                       df=m*(n-1),
                                       nc=0)*(scale**-1)
            scale = sig2_curr/(m*(n-1))
            fs2_curr = stats.ncx2._pdf(hat_sig2*(scale**-1),
                                       df=m*(n-1),
                                       nc=0)*(scale**-1)

            # take ratio of two current and candidate likelihoods
            a = (fs2_cand*fd2x_cand*fd2y_cand)/(fs2_curr*fd2x_curr*fd2y_curr)
            if a >= 1:
                # current becomes candidate
                sig2_curr = sig2_cand
                d2x_curr = d2x_cand
                d2y_curr = d2y_cand
                accept[i] = 1
            else:
                if u[i] < a:
                    # current becomes candidate
                    accept[i] = 1
                    sig2_curr = sig2_cand
                    d2x_curr = d2x_cand
                    d2y_curr = d2y_cand
        # add current to trace.
        trace[:, i] = [d2x_curr, d2y_curr, sig2_curr]
    accept_p = np.mean(accept)
    return trace, accept_p


def get_emp_dist_r2er_n2n(r2er_check, trace, m, n, n_r2er_sims=100):
    """Sample from hat_r^2_{ER} with draws from poster sig2, d2x d2y trace."""
    sig2_post = trace[2]  # trial-to-trial variance
    d2xm_post = trace[0]*m  # dynamic range
    d2ym_post = trace[1]*m  # dynamic range

    # randomly sample from post-dist
    sample_inds = np.random.choice(len(sig2_post),
                                   size=n_r2er_sims,
                                   replace=True)

    res = np.zeros(n_r2er_sims)
    for j in range(n_r2er_sims):
        k = sample_inds[j]  # get random indices for posterior samples
        x, y = sim_n2n(r2er=r2er_check,
                       sig2=sig2_post[k],
                       d2x=d2xm_post[k],
                       d2y=d2ym_post[k],
                       n=n,
                       m=m,
                       n_exps=1,
                       verbose=True)

        res[j] = (r2er_n2n(x, y)[0]).squeeze()  # calculate hat_r2er

    return res


def find_sgn_p_cand_n2n(r2er_cand, r2er_hat_obs, alpha_targ, trace, m, n,
                        p_thresh=0.01, n_r2er_sims=1000):
    """Find if value of cdf of r2er_cand at r2er_hat_obs is >, <, or == alpha.

    helper function for find_r2er_w_quantile_n2n
    """
    z_thresh = stats.norm.ppf(1.-p_thresh)
    res = get_emp_dist_r2er_n2n(r2er_cand, trace, m, n,
                                n_r2er_sims=n_r2er_sims)

    count = (res < r2er_hat_obs).sum()

    # z-test
    z = ((count - alpha_targ*n_r2er_sims) /
         (n_r2er_sims*alpha_targ*(1-alpha_targ))**0.5)

    if z > z_thresh:  # signif. above desired count
        sgn_p_cand = 1
    elif -z > z_thresh:  # signif. below desired count
        sgn_p_cand = -1
    else:  # NS different
        sgn_p_cand = 0
    return sgn_p_cand, res


def find_r2er_w_quantile_n2n(r2er_hat_obs, alpha_targ, trace, m, n, n_splits=6,
                             p_thresh=1e-2, n_r2er_sims=100, int_l=0, int_h=1):
    """Find r2er with r2er_hat distribution  CDF(r2er_hat_obs)=alpha_targ.

    Helper function for ecci_r2er_n2n.
    """
    # we first check at the highest and lowest possible values
    sgn_p_cand_h, res = find_sgn_p_cand_n2n(r2er_cand=int_h,
                                            r2er_hat_obs=r2er_hat_obs,
                                            alpha_targ=alpha_targ,
                                            trace=trace, m=m, n=n,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims)

    sgn_p_cand_l, res = find_sgn_p_cand_n2n(r2er_cand=int_l,
                                            r2er_hat_obs=r2er_hat_obs,
                                            alpha_targ=alpha_targ,
                                            trace=trace, m=m, n=n,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims)

    # if the highest possible value is too low, or is NS different from
    # desired alpha return it as the r2_er with the desired alpha
    if sgn_p_cand_h == 1 or sgn_p_cand_h == 0:
        return int_h, res

    # if the lowest possible value is too high, or is NS different
    # from desired alpha return it as the r2_er with the desired alpha
    if sgn_p_cand_l == -1 or sgn_p_cand_l == 0:
        return int_l, res

    # if endpoints are not r2_er desired then begin iterative bracketing
    # with max of n_splits iterations.
    for split in range(n_splits):
        # choose random point in current r2_Er range
        c_cand = np.random.uniform(int_l, int_h)

        # evaluate it
        sgn_p_cand, res = find_sgn_p_cand_n2n(r2er_cand=c_cand,
                                              r2er_hat_obs=r2er_hat_obs,
                                              alpha_targ=alpha_targ,
                                              trace=trace, m=m, n=n,
                                              p_thresh=p_thresh,
                                              n_r2er_sims=n_r2er_sims)
        # if it was too high then set it as the  new upper end of the interval
        if sgn_p_cand == -1:
            int_h = c_cand
        # if it was too low then set it as the new lower end
        elif sgn_p_cand == 1:
            int_l = c_cand
        # if it was NS diffent then return the candidate
        else:
            return c_cand, res

    return c_cand, res


def ecci_r2er_n2n(x, y, alpha_targ=0.1, n_r2er_sims=1000,  p_thresh=0.01,
                  n_splits=6, trace=None):
    """Find alpha_targ level confidence intervals for hat_r2_{ER}.

    Parameters
    ----------
    x : numpy.ndarray
        m observations model predictions
    y : numpy.ndarray
        n repeats X m observations array of data
    alpha_targ : float
        desired alpha level (0,1) (proportion of CI's containing r^2_er)
    n_r2er_sims : int
        how many samples of n_r2er_sims hat_r2_{ER} to draw to calculate
        quantile estimates and how long trace should be
    p_thresh: float
        p-value below which we will reject null hypothesis that a candidate
        r2_ER gives alpha_targ
     n_splits : int
         for iterative bracketing algorithm find_cdf_pos the number of times it
         will split so 6 times gives a range of potential value ~2**-6 = 0.015
         in length.
    trace : numpy.ndarray
        if trace for posterior has already been calculated you can use it here
        if none then finds posterior.

    Returns
    -------
    ll : float
        lower end of confidence interval
    ul : float
        upper end of confidence interval
    r2er_hat_obs : float
        estimate hat_r2_{er}

    trace : numpy.ndarray
        posterior trace

    ll_alpha : numpy.ndarray
        the distribution of hat_r^2_er assosciated with r^2_er(l)

    ul_alpha : numpy.ndarray
        samples from hat_r^2_er assosciated with r^2_er(h)

    --------
    """
    # get confidence intervals
    n, m = y.shape
    r2er_hat_obs = r2er_n2n(x, y)[0]
    # calculate m_traces different traces so that converge
    # stat gelmin rubin rhat can be calculated
    if trace is None:
        m_traces = 5
        traces = np.zeros((m_traces, 3, n_r2er_sims))
        for a_m in range(m_traces):

            trace, p = sample_post_s2_d2x_d2y(
                y, x, n_samps=n_r2er_sims)  # get posterior dist of params
            traces[a_m] = trace
        for i in range(2):
            rhat = gr_rhat(traces[:, i].T)
            if rhat > 1.2:
                print(('warning r_hat=' + str(np.round(rhat, 2)) +
                       ' is greater than 1.2, try increasing n_r2er_sims'))

    ul, ul_alpha = find_r2er_w_quantile_n2n(r2er_hat_obs, alpha_targ/2,
                                            trace, m, n,
                                            n_splits=n_splits,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims,
                                            int_l=0, int_h=1)
    ll, ll_alpha = find_r2er_w_quantile_n2n(r2er_hat_obs, 1-alpha_targ/2,
                                            trace, m, n,
                                            n_splits=n_splits,
                                            p_thresh=p_thresh,
                                            n_r2er_sims=n_r2er_sims,
                                            int_l=0, int_h=1)

    return ll, ul, r2er_hat_obs, trace, ll_alpha, ul_alpha
