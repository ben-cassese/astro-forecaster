###############################################################################
# Imports
###############################################################################
import numpy as np
from scipy.stats import norm
from scipy.special import erfc
import h5py
from astropy import constants as const
import pkg_resources
from warnings import simplefilter, warn
from joblib import Parallel, delayed


###############################################################################
# Setup
###############################################################################
simplefilter('always', UserWarning)
hyper_file = pkg_resources.resource_filename('forecaster', 'fitting_parameters.h5')
h5 = h5py.File(hyper_file, 'r')
all_hyper = h5['hyper_posterior'][:]
h5.close()
n_pop = 4



###############################################################################
# Mass to Radius
###############################################################################

def Mpost2R(mass_array, unit='Jupiter', classify=False, guardrails=True):
    """
    Description:
    ---------------
	Given an array of masses, return an equal length
        array of forecasted radii. Masses/radii do not
        have to correspond to a single physical source:
        output indecies correspond to input indecies,
        so the mass array can hold information for
        multiple objects.

    Parameters:
    ---------------
	mass_array: one dimensional array
		Input mass array
	unit: str (optional)
		Unit of mass_array.
        Default is 'Jupiter'.
		Options are 'Earth' and 'Jupiter'.
	classify: boolean (optional)
		Indicator for whether to calculate the probabilities that
		the mass array represents each object in
		{Terran, Neptunian, Jovian, Stellar}.
		Default is False.
		Do not use if the mass array does not represent a single
		object.

    Returns
    ---------------
	If classify==False:
		radii: one dimensional array
			Predicted radius distribution in the specified input unit
	Else:
		radii: one dimensional array
			Predicted radius distribution in the specified input unit
		classification: dict
			Probabilities the mass array represents an object
			in each of the 4 populations.

	"""
    def internal(mass_array, unit, classify):
        # Initial setup
        assert len(mass_array.shape) == 1 and len(mass_array) > 0, \
            "Input mass must 1-D array with non-zero length"
        assert len(mass_array) <= 1e6, \
            "Input mass array must have a length < 1e6"
        sample_size = len(mass_array)
        hyper_ind = np.random.randint(low = 0,
                                          high = np.shape(all_hyper)[0],
                                          size = sample_size)
        hypers = all_hyper[hyper_ind,:]
        Probs = np.random.random(sample_size)


        # Convert internally to Earth masses
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            mass_array = mass_array * (const.M_jup / const.M_earth).value
        else:
            unit = 'Jupiter'
            mass_array = mass_array * (const.M_jup / const.M_earth).value
            warn("Input unit must be 'Earth' or 'Jupiter'. " + \
                  "Using 'Jupiter' as default.")


        # Ensure input within model expectations
        if guardrails:
            if np.sum(mass_array > 3e5) > 0:
                raise ValueError('Mass array contains values above 3e5 M_e, ' + \
                         'outside of model expectation. Returning None')
            if np.sum(mass_array < 3e-4) > 0:
                raise ValueError('Mass array contains values below 3e-4 M_e, ' + \
                         'outside of model expectation. Returning None')


        logMs = np.log10(mass_array)

        # Get the data needed to calculate radii
        w = split_hyper_linear(hypers)
        # w, read as (#hyper, flag for {C, slope, sigma, trans}, pop #)
        TS = np.zeros((len(w), 5))*np.nan
        TS[:, 0] = -np.inf
        TS[:, 1:4] = w[:, -1, :-1]
        TS[:, -1] = np.inf

        pop_num = np.zeros(len(logMs))*np.nan
        pop_num[((logMs > TS[:,0]) & (logMs < TS[:,1]))] = 0
        pop_num[((logMs > TS[:,1]) & (logMs < TS[:,2]))] = 1
        pop_num[((logMs > TS[:,2]) & (logMs < TS[:,3]))] = 2
        pop_num[((logMs > TS[:,3]) & (logMs < TS[:,4]))] = 3
        pop_num = pop_num.astype(int)

        Cs = w[np.arange(0,len(w),1), 0, pop_num]
        Slopes = w[np.arange(0,len(w),1), 1, pop_num]
        Mus = Cs + logMs*Slopes
        Sigs = w[np.arange(0,len(w),1), 2, pop_num]


        # Calculate the radii
        logRs = norm.ppf(Probs, Mus, Sigs)
        radii_sample = 10.** logRs


        # convert to right unit
        if unit == 'Jupiter':
            radii = radii_sample / (const.R_jup / const.R_earth).value
        else:
            radii = radii_sample

       # Return
        if classify:
            prob = np.zeros(4)
            prob[0] = np.sum(pop_num == 0)
            prob[1] = np.sum(pop_num == 1)
            prob[2] = np.sum(pop_num == 2)
            prob[3] = np.sum(pop_num == 3)
            prob = prob / np.sum(prob) * 100
            return radii, {'Terran': prob[0],
                           'Neptunian': prob[1],
                           'Jovian': prob[2],
                           'Stellar': prob[3]}
        else:
            return radii

    if len(mass_array) > 1e6:
        warn("Large input array, breaking into smaller chunks to run in parallel")
        mass_chunks = []
        if len(mass_array) % int(1e6) > 0:
            r = np.floor(len(mass_array) / 1e6).astype(int) + 1
        else:
            r = np.floor(len(mass_array) / 1e6).astype(int)
        for i in range(r):
            mass_chunks.append(mass_array[i*int(1e6):min((i+1)*int(1e6),
                                                         len(mass_array))])
        q = Parallel(n_jobs=-1, prefer="threads")(delayed(internal)
                                (chunk, unit, classify) for chunk in mass_chunks)
        if not classify:
            radii = np.hstack(q)
            return radii
        else:
            radii = np.array([])
            for chunk in q:
                radii = np.concatenate((radii, chunk[0]))

            classifications = np.zeros((len(q), 4))
            for i in range(len(q)):
                classifications[i] = np.array(list(q[i][1].values()))*len(q[i][0])
            z = dict(zip(q[0][1].keys(),
                    np.sum(classifications, axis=0) / np.sum(classifications)*100))

            return radii, z
    else:
        return internal(mass_array, unit, classify)

#------------------------------------------------------------------------------

def Mstat2R(mean, onesig_neg, onesig_pos,
            unit='Jupiter',
            n_mass_samples=int(1e3),
            classify=False):
    """
    Description:
    ---------------
	Given a mean and (possibly asymmetric) uncertainties in mass,
        return the median and +/- one sigma uncertainties in radius.
        Relies on Mpost2R and draw_from_asymmetric

    Parameters:
    ---------------
	mean: float
		Input mass
	onesig_neg: float
		The one sigma negative uncertainty
	onesig_pos: float
		The one sigma positive uncertainty
	unit: str (optional)
		Unit of the input radius.
		Default is 'Jupiter'.
		Options are 'Earth' and 'Jupiter'.
	n_mass_samples: int (optional)
		The number of draws from a distribution created using
		stated mean and uncertainties.
		Default is 1000
	classify: boolean (optional)
		Indicator for whether to calculate the probabilities that
		the input mass represents each object in
		{Terran, Neptunian, Jovian, Stellar}.
		Default is False.


    Returns
    ---------------
	If classify==False:
		radius: a (3,) array
			Values are [median radius, + uncertainty, - uncertainty],
			All values in the specified input units
	Else:
		radius: a (3,) array
			Values are [median radius, + uncertainty, - uncertainty],
			All values in the specified input units
		classification: dict
			Probabilities the input mass statistics represent an object
			in each of the 4 populations.

	"""
    # Initial setup
    onesig_neg = np.abs(onesig_neg)
    onesig_pos = np.abs(onesig_pos)
    if onesig_neg == 0:
        warn("Negative uncertainty cannot be zero, using 1e-9 instead")
        onesig_neg = 1e-9
    if onesig_pos == 0:
        warn("Positive uncertainty cannot be zero, using 1e-9 instead")
        onesig_pos = 1e-9


    # Create an array of masses from the given statistics
    masses = draw_from_asymmetric(mu=mean,
                                  signeg=onesig_neg,
                                  sigpos=onesig_pos,
                                  xmin=0, xmax=np.inf,
                                  nsamples=n_mass_samples)

    # Convert that mass array to radius
    r = Mpost2R(masses, unit=unit, classify=classify)

    # Address the different shaped outputs depending on classify
    if classify:
        radii = r[0]
    else:
        radii = r

    # Calculate the returned statistics
    med = np.median(radii)
    onesigma = 34.1
    stats =  np.array([med, np.percentile(radii, 50.+onesigma, \
                                       interpolation='nearest') - med,
                       -(med - np.percentile(radii, 50.-onesigma, \
                                       interpolation='nearest'))])


    if classify:
        return stats, r[1]
    else:
        return stats



###############################################################################
# Radius to Mass
###############################################################################

def Rpost2M(radius_array, unit='Jupiter', grid_size=int(1e3), classify=False, guardrails=True):
    """
    Description:
    ---------------
	Given an array of radii, return an equal length
        array of forecasted masses. Masses/radii do not
        have to correspond to a single physical source:
        output indecies correspond to input indecies,
        so the mass array can hold information for
        multiple objects.

    Parameters:
    ---------------
	radius_array: one dimensional array
		Input radius array
	unit: str (optional)
		Unit of radius_array.
        Default is 'Jupiter'.
		Options are 'Earth' and 'Jupiter'.
	grid_size: int
		The size of the possible masses considered
		in the range [-3.522, 5.477] log(M_e).
		Default is 1e3.
		Anything below 10 will be converted to 10.
	classify: boolean (optional)
		Indicator for whether to calculate the probabilities that
		the radius array represents each object in
		{Terran, Neptunian, Jovian, Stellar}.
		Default is False.
		Do not use if the mass array does not represent a single
		object.

    Returns
    ---------------
	If classify==False:
		mass: one dimensional array
			Predicted mass distribution in the specified input unit
	Else:
		mass: one dimensional array
			Predicted mass distribution in the specified input unit
		classification: dict
			Probabilities the radius array represents an object
			in each of the 4 populations.

	"""
    def internal(radius_array, unit, grid_size, classify):
        # Initial setup
        assert len(radius_array.shape) == 1 and len(radius_array) > 0, \
            "Input radius must 1-D array with non-zero length"
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            radius_array = radius_array * (const.R_jup / const.R_earth).value
        else:
            unit = 'Jupiter'
            radius_array = radius_array * (const.R_jup / const.R_earth).value
            warn("Input unit must be 'Earth' or 'Jupiter'. " + \
                  "Using 'Jupiter' as default.")

        # Ensure sample grid isn't too sparse
        if grid_size < 10:
            Warn('The sample grid of masses is too sparse, replacing ' +\
                 'grid_size with 10 instead.')
            grid_size = 10

        # Ensure input within model expectations
        if guardrails:
            if np.sum(radius_array > 1e2) > 0:
                raise ValueError('Radius array contains values above 1e2 R_e, ' + \
                         'outside of model expectation. Returning None')
            if np.sum(radius_array < 1e-1) > 0:
                raise ValueError('Mass array contains values below 1e-1 M_e, ' + \
                         'outside of model expectation. Returning None')

        # Get the data to convert to masses
        sample_size = len(radius_array)
        logr = np.log10(radius_array)
        logm_grid = np.linspace(-3.522, 5.477, grid_size)
        hyper_ind = np.random.randint(low = 0,
                                      high = np.shape(all_hyper)[0],
                                      size = sample_size)
        hypers = all_hyper[hyper_ind,:]
        w = split_hyper_linear(hypers)
        Ind = indicate(logm_grid, w)
        Probs = ProbRGivenM(log_radii=logr, M=logm_grid,
                            indicate_output=Ind, split_hyper_linear_output=w)

        # Calculate the masses
        logm = logm_grid[(Probs.cumsum(1) > \
                          np.random.rand(Probs.shape[0])[:,None]).argmax(1)]
        mass_sample = 10.** logm

        # Convert to original unit
        if unit == 'Jupiter':
            mass = mass_sample / (const.M_jup / const.M_earth).value
        else:
            mass = mass_sample

        if classify:
            ind = indicate(logm, w)
            prob = np.sum(np.sum(ind, axis=2), axis=0) / \
                        (len(logm) * len(radius_array)) * 100
            return mass, {'Terran': prob[0],
                           'Neptunian': prob[1],
                           'Jovian': prob[2],
                           'Stellar': prob[3]}
        else:
            return mass

    if len(radius_array) > 500:
        warn("Large input array, breaking into smaller chunks to run in parallel")
        radius_chunks = []
        if len(radius_array) % int(500) > 0:
            r = np.floor(len(radius_array) / 500).astype(int) + 1
        else:
            r = np.floor(len(radius_array) / 500).astype(int)
        for i in range(r):
            radius_chunks.append(radius_array[i*int(500):min((i+1)*int(500),
                                                         len(radius_array))])
        q = Parallel(n_jobs=-1, prefer="threads")(delayed(internal)
                                (chunk, unit, grid_size, classify) for chunk in radius_chunks)
        if not classify:
            masses = np.hstack(q)
            return masses
        else:
            masses = np.array([])
            for chunk in q:
                masses = np.concatenate((masses, chunk[0]))

            classifications = np.zeros((len(q), 4))
            for i in range(len(q)):
                classifications[i] = np.array(list(q[i][1].values()))*len(q[i][0])

            z = dict(zip(q[0][1].keys(),
                    np.sum(classifications, axis=0) / np.sum(classifications)*100))

            return masses, z
    else:
        return internal(radius_array, unit, grid_size, classify)

#------------------------------------------------------------------------------

def Rstat2M(mean, onesig_neg, onesig_pos,
            unit='Jupiter', n_radii_samples=int(1e3),
            mass_grid_size=int(1e3), classify=False):
    """
    Description:
    ---------------
	Given a mean and (possibly asymmetric) uncertainties in radius,
        return the median and +/- one sigma uncertainties in mass.
        Relies on Rpost2M and draw_from_asymmetric

    Parameters:
    ---------------
	mean: float
		Input mass
	onesig_neg: float
		The one sigma negative uncertainty
	onesig_pos: float
		The one sigma positive uncertainty
	unit: str (optional)
		Unit of the input mass.
		Default is 'Jupiter'.
		Options are 'Earth' and 'Jupiter'.
	n_radii_samples: int (optional)
		The number of draws from a distribution created using
		stated mean and uncertainties.
		Default is 1000
	mass_grid_size: int (optional)
		The size of the mass grid considered in Rpost2M
	classify: boolean (optional)
		Indicator for whether to calculate the probabilities that
		the input mass represents each object in
		{Terran, Neptunian, Jovian, Stellar}.
		Default is False.


    Returns
    ---------------
	If classify==False:
		mass: a (3,) array
			Values are [median mass, + uncertainty, - uncertainty],
			All values in the specified input units
	Else:
		mass: a (3,) array
			Values are [median mass, + uncertainty, - uncertainty],
			All values in the specified input units
		classification: dict
			Probabilities the input radius statistics represent an object
			in each of the 4 populations.

	"""
    # Initial setup
    onesig_neg = np.abs(onesig_neg)
    onesig_pos = np.abs(onesig_pos)
    if onesig_neg == 0:
        warn("Negative uncertainty cannot be zero, using 1e-9 instead")
        onesig_neg = 1e-9
    if onesig_pos == 0:
        warn("Positive uncertainty cannot be zero, using 1e-9 instead")
        onesig_pos = 1e-9

    radii = draw_from_asymmetric(mu=mean,
                                 signeg=onesig_neg,
                                 sigpos=onesig_pos,
                                 xmin=0, xmax=np.inf,
                                 nsamples=n_radii_samples)

    m = Rpost2M(radii, unit=unit,
                grid_size=mass_grid_size,
                classify=classify)

    if classify:
        masses = m[0]
    else:
        masses = m

    med = np.median(masses)
    onesigma = 34.1
    stats =  np.array([med, np.percentile(masses, 50.+onesigma, \
                                       interpolation='nearest') - med,
                       -(med - np.percentile(masses, 50.-onesigma, \
                                       interpolation='nearest'))])


    if classify:
        return stats, m[1]
    else:
        return stats



###############################################################################
# Helper Functions
###############################################################################

def draw_from_asymmetric(mu, signeg, sigpos, xmin, xmax, nsamples):
    '''
    Implement an asymmetric distribution sampling method
    written for a Mathematica notebook in Python.
    Original source: https://github.com/davidkipping/asymmetric
    This breaks when signeg or sigpos = 0, so replace those with
    a tiny value if they are actually valid
    '''

    signeg = np.abs(signeg)
    sigpos = np.abs(sigpos)
    Xs = np.random.uniform(mu-20*signeg, mu+20*sigpos, 1000000)
    if (np.max(Xs) > xmax) and (np.min(Xs) > xmin):
        Xs = np.random.uniform(mu-20*signeg, xmax, 1000000)
    elif (np.max(Xs) < xmax) and (np.min(Xs) < xmin):
        Xs = np.random.uniform(xmin, mu+20*sigpos, 1000000)
    elif (np.max(Xs) < xmax) and (np.min(Xs) > xmin):
        Xs = np.random.uniform(mu-20*signeg, mu+20*sigpos, 1000000)
    elif (np.max(Xs) > xmax) and (np.min(Xs) < xmin):
        Xs = np.random.uniform(xmin, xmax, 1000000)
    pdf = np.zeros(len(Xs))*np.nan

    pdf[Xs < mu] = np.exp(- (Xs[Xs < mu] - mu)**2 / (2*signeg**2)) / \
            (2*np.pi*signeg*sigpos * \
            (1 / (np.sqrt(2*np.pi) * signeg) + \
             1 / (np.sqrt(2*np.pi) * sigpos)) * \
            (0.5 - 0.5*erfc((mu - xmin) / (np.sqrt(2) * signeg))))

    pdf[Xs >= mu] = np.exp(- (Xs[Xs >= mu] - mu)**2 / (2*sigpos**2)) / \
            (2*np.pi*signeg*sigpos * \
            (1 / (np.sqrt(2*np.pi) * signeg) + \
             1 / (np.sqrt(2*np.pi) * sigpos)) * \
            (0.5*erfc((mu - xmax) / (np.sqrt(2) * sigpos)) - 0.5))

    pdf = pdf/np.sum(pdf)

    v = np.random.choice(Xs, nsamples, p=pdf)
    return v


def split_hyper_linear(hypers):
    '''
    Convert the raw output of a selection from the
    hyperparameter file into something useable later
    in processing. Vectorized version of the original
    split_hyper_linear in chenjj2's forecaster.
    '''
    C0 = hypers[:, 0]
    Slope = hypers[:, 1:1+n_pop]
    Sigma = hypers[:, 1+n_pop:1+2*n_pop]
    Trans = hypers[:, 1+2*n_pop:]

    C = np.zeros_like(Slope)
    C[:,0] = C0
    C[:,1] = C[:,0] + Trans[:,0] * (Slope[:,0] - Slope[:,1])
    C[:,2] = C[:,1] + Trans[:,1] * (Slope[:,1] - Slope[:,2])
    C[:,3] = C[:,2] + Trans[:,2] * (Slope[:,2] - Slope[:,3])

    # Read as (#hyper/radii, flag for {C, slope, sigma, trans}, pop #)
    # Trans only has 3 numbers, leave the last as nan
    w = np.zeros((len(hypers), 4, 4))*np.nan
    w[:, 0, :] = C
    w[:, 1, :] = Slope
    w[:, 2, :] = Sigma
    w[:, 3, :-1] = Trans

    return w


def indicate(logM, split_hyper_output):
    '''
    Flag which population given associated masses and
    hyperparameter belong to. Vectorized version of
    the original indicate in chenjj2's forecaster.
    '''
    TS = np.zeros((len(split_hyper_output), 5))*np.nan
    TS[:, 0] = -np.inf
    TS[:, 1:4] = split_hyper_output[:, -1, :-1]
    TS[:, -1] = np.inf
    TS

    # Read as Ind[hyper/radius, pop#, mass grid spot]
    Ind = np.zeros((len(split_hyper_output), 4, len(logM)))
    Ind[:, 0, :] = ((logM[:, np.newaxis] >= TS[:, 0]) & (logM[:, np.newaxis]  < TS[:, 1])).T
    Ind[:, 1, :] = ((logM[:, np.newaxis] >= TS[:, 1]) & (logM[:, np.newaxis]  < TS[:, 2])).T
    Ind[:, 2, :] = ((logM[:, np.newaxis] >= TS[:, 2]) & (logM[:, np.newaxis]  < TS[:, 3])).T
    Ind[:, 3, :] = ((logM[:, np.newaxis] >= TS[:, 3]) & (logM[:, np.newaxis]  < TS[:, 4])).T
    Ind =  Ind.astype(bool)
    return Ind


def ProbRGivenM(log_radii, M, indicate_output, split_hyper_linear_output):
    '''
    For each input mass, calculate the probability
    of that object corresponding to each of the input radii.
    Vectorized version of
    the original ProbRGivenM in chenjj2's forecaster.
    '''
    w = split_hyper_linear_output
    Ind = indicate_output
    Mexpanded = np.ones((len(log_radii), len(M)))*M
    Mu = np.zeros((len(log_radii), len(M)))*np.nan
    Mu[Ind[:,0,:]] = (w[:, 0, 0][:, np.newaxis] + (Mexpanded*w[:, 1, 0][:, np.newaxis]))[Ind[:,0,:]]
    Mu[Ind[:,1,:]] = (w[:, 0, 1][:, np.newaxis] + (Mexpanded*w[:, 1, 1][:, np.newaxis]))[Ind[:,1,:]]
    Mu[Ind[:,2,:]] = (w[:, 0, 2][:, np.newaxis] + (Mexpanded*w[:, 1, 2][:, np.newaxis]))[Ind[:,2,:]]
    Mu[Ind[:,3,:]] = (w[:, 0, 3][:, np.newaxis] + (Mexpanded*w[:, 1, 3][:, np.newaxis]))[Ind[:,3,:]]


    Sig = np.zeros((len(log_radii), len(M)))*np.nan
    Sig[Ind[:,0,:]] = (np.ones((len(log_radii), len(M)))*w[:, 2, 0][:, np.newaxis])[Ind[:,0,:]]
    Sig[Ind[:,1,:]] = (np.ones((len(log_radii), len(M)))*w[:, 2, 1][:, np.newaxis])[Ind[:,1,:]]
    Sig[Ind[:,2,:]] = (np.ones((len(log_radii), len(M)))*w[:, 2, 2][:, np.newaxis])[Ind[:,2,:]]
    Sig[Ind[:,3,:]] = (np.ones((len(log_radii), len(M)))*w[:, 2, 3][:, np.newaxis])[Ind[:,3,:]]

    Probs = norm.pdf(x=np.repeat(log_radii, len(M)),
                     loc=Mu.reshape(len(M)*len(log_radii)),
                     scale=Sig.reshape(len(M)*len(log_radii)))
    Probs = Probs.reshape((len(log_radii), len(M)))
    Probs = Probs / np.sum(Probs, axis=1)[:, np.newaxis]

    return Probs
