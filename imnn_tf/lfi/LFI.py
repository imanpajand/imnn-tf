"""Approximate Bayesian computation with IMNN

This module provides the methods necessary to perform various ABC methods using the IMNN.

TODO
____
The documentation for this module is not complete, and stability may be patchy (hopefully not). If you find any problems please push an issue to the GitHub.
"""


__version__ = '0.2.0'
__author__ = "Tom Charnock"


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
from imnn_tf.utils import utils
import tqdm
np.set_printoptions(precision=3, suppress=True)

class LFI():
    """

    Attributes
    __________
    prior : class
        the truncated Gaussian priors to draw parameters values from
    fisher : ndarray
        Fisher information matrix calculated from last run summaries
    get_estimate : func
        get estimate from network
    simulator : func
        single input lambda function of the simulator
    n_params : int
        the number of parameters in the model
        the number of total draws from the proposal for the PMC
    """
    def __init__(self, target_data, prior, Fisher, get_estimate, simulator, labels=None):
        """Initialises the ABC class and calculates some useful values

        Parameters
        __________
        target_data : ndarray
            the observed data. in principle several observations can be passed
            at one time.
        prior : class
            the truncated Gaussian priors to draw parameters values from
        Fisher : TF tensor float (n_params, n_params)
            approximate Fisher information to use for ABC
        get_estimate : func
            function for obtaining estimate from neural network
        simulator : func
            single input lambda function of the simulator
        """
        self.utils = utils()
        self.prior = prior
        event_shape = self.prior.event_shape
        if len(event_shape) == 0:
            print("`prior.event_shape` must be at least `[1]`")
            sys.exit()
        else:
            self.n_params = event_shape[0]
        if not hasattr(self.prior, "low"):
            low = []
            if hasattr(self.prior, "distributions"):
                for distribution in self.prior.distributions:
                    if hasattr(distribution, "low"):
                        low.append(distribution.low.numpy())
                    else:
                        low.append(-np.inf)
            else:
                low = [-np.inf for i in range(self.n_params)]

            self.prior.low = low
        if not hasattr(self.prior, "high"):
            high = []
            if hasattr(self.prior, "distributions"):
                for distribution in self.prior.distributions:
                    if hasattr(distribution, "high"):
                        high.append(distribution.high.numpy())
                    else:
                        high.append(np.inf)
            else:
                high = [np.inf for i in self.n_params]
            self.prior.high = high
        if Fisher is not None:
            if type(Fisher) == type(tf.constant(0)):
                self.F = Fisher.numpy()
            else:
                self.F = Fisher
            self.Finv = np.linalg.inv(self.F)
        else:
            self.F = None
            self.Finv = None
        self.data = target_data
        if get_estimate is not None:
            estimate = get_estimate(self.data)
            if type(estimate) == type(tf.constant(0)):
                self.estimate = estimate.numpy()
                self.get_estimate = lambda x : get_estimate(x).numpy()
            else:
                self.estimate = estimate
                self.get_estimate = get_estimate
            self.targets = self.estimate.shape[0]
        else:
            self.estimate = None
            self.get_estimate = None
            self.targets = None
        self.labels = labels
        self.simulator = simulator

    def scatter(self, indices, updates, shape):
        a = np.zeros(shape)
        np.add.at(
            a,
            tuple(indices),
            updates)
        return a

    def levels(self, array, levels):
        array = np.sort(array.flatten())[::-1]
        cdf = np.cumsum(array / np.sum(array))
        if type(levels) == list:
            contours = []
            for level in levels:
                contours.append(array[np.argmin(np.abs(cdf - level))])
            contours = np.unique(contours)
        else:
            contours = array[np.argmin(np.abs(cdf - levels))]
        return contours

    def plot_Fisher(self, ax=None, figsize=(10, 10), save=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        a = ax.imshow(self.Finv, extent=[0, self.n_params, 0, self.n_params])
        ax.set_title("Inverse Fisher")
        temp_labels = ["" for i in range(2 * self.n_params + 1) ]
        if self.labels is not None:
            ax.set_xticks(ticks=[i + 0.5 for i in range(self.n_params)])
            ax.set_xticklabels(labels=self.labels)
            ax.set_yticks(ticks=[i + 0.5 for i in range(self.n_params)])
            ax.set_yticklabels(labels=self.labels)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Parameters")
        plt.colorbar(a, ax=ax, fraction=0.046, pad=0.04)
        if save is not None:
            plt.savefig(save,
                        bbox_inches="tight",
                        transparancy=True)
        return ax

    def setup_triangle_plot(self, ax=None, figsize=None, wspace=0.1, hspace=0.1, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(self.n_params, self.n_params, figsize=figsize)
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        for plot in range(self.n_params**2):
            i_ = plot % self.n_params
            j_ = plot // self.n_params
            if i_ < j_:
                if i_ == 0:
                    if self.labels is not None:
                        ax[j_, i_].set_ylabel(self.labels[j_])
                if j_ == self.n_params - 1:
                    if self.labels is not None:
                        ax[j_, i_].set_xlabel(self.labels[i_])
                if j_ < self.n_params - 1:
                    ax[j_, i_].set_xticks([])
                if i_ > 0:
                    ax[j_, 0].get_shared_y_axes().join(
                    ax[j_, 0], ax[j_, i_])
                    ax[j_, i_].set_yticks([])
                if j_ > 0:
                    ax[0, i_].get_shared_x_axes().join(
                    ax[0, i_], ax[j_, i_])
            elif i_ == j_:
                ax[i_, j_].yaxis.tick_right()
                if self.labels is not None:
                    ax[j_, i_].set_ylabel(
                        r"$\mathcal{P}($" + self.labels[i_] + "$|{\\bf t})$",
                        rotation=270,
                        labelpad=15)
                    ax[j_, i_].yaxis.set_label_position("right")
                if j_ < self.n_params - 1:
                    ax[i_, j_].set_xticks([])
                if j_ == self.n_params - 1:
                    if self.labels is not None:
                        ax[j_, i_].set_xlabel(self.labels[i_])
                if j_ > 0:
                    ax[0, i_].get_shared_x_axes().join(
                    ax[0, i_], ax[j_, i_])
            else:
                ax[j_, i_].axis("off")
        return ax

    def triangle_plot(self, distribution, grid, meshgrid=True, color=None,
                      label=None, levels=[0.68, 0.95, 0.99], smoothing=None,
                      **kwargs):
        if smoothing is not None:
            smoother = lambda x : gaussian_filter(x, smoothing, mode="nearest")
        else:
            smoother = lambda x : x
        if meshgrid:
            grid = np.array(
                [np.mean(
                    grid[i],
                    axis=tuple(
                        np.setdiff1d(
                            np.arange(
                                self.n_params),
                            i)))
                 for i in range(self.n_params)])
        if len(distribution.shape) == self.n_params:
            distribution = distribution[np.newaxis, ...]
        ax = self.setup_triangle_plot(**kwargs)
        colours = []
        for plot in range(self.n_params**2):
            i_ = plot % self.n_params
            j_ = plot // self.n_params
            if i_ == j_:
                for datum in range(distribution.shape[0]):
                    this_distribution = smoother(
                        np.sum(
                            distribution[datum],
                            axis=tuple(
                                np.setdiff1d(
                                    np.arange(self.n_params),
                                    i_))))
                    this_distribution = this_distribution / np.sum(
                        this_distribution *
                        (grid[i_][1] - grid[i_][0]))
                    a, = ax[j_, i_].plot(grid[i_], this_distribution.T,
                                         color=color, label=label)
                    colours.append(a.get_color())
                if i_ == 0:
                    if label is not None:
                        ax[j_, i_].legend(frameon=False, loc=2)
            if i_ < j_:
                for datum in range(distribution.shape[0]):
                    this_distribution = smoother(
                        np.sum(
                            distribution[datum],
                            axis=tuple(
                                np.setdiff1d(
                                    np.arange(self.n_params),
                                    np.array([i_, j_])))))
                    this_distribution = this_distribution / np.sum(
                        this_distribution *
                        (grid[i_][1] - grid[i_][0]) *
                        (grid[j_][1] - grid[j_][0]))
                    ax[j_, i_].contour(
                        grid[i_],
                        grid[j_],
                        this_distribution.T,
                        colors=colours[datum],
                        levels=self.levels(this_distribution, levels))
        return ax

class ApproximateBayesianComputation(LFI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def __call__(self, draws, at_once=True, save_sims=None, predrawn=False):
        return self.ABC(draws, at_once, save_sims=None, predrawn=predrawn)

    def ABC(self, draws, at_once=True, save_sims=None, predrawn=False, update=True):
        """Approximate Bayesian computation

        Here we draw some parameter values from the prior supplied to the class
        and generate simulations. We then use the IMNN to compress the sims
        into summaries and compare those to the summary of the observed data.

        All summaries are collected so that the acceptance epsilon can be
        modified at the users will.

        Parameters
        __________
        draws : int
            number of parameter draws to make, this number of simulations will be run.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        return_dict : bool, optional
            the ABC_dict attribute is normally updated, but the dictionary can
            be returned by the function. this is used by the PMC.
        predrawn : bool, optional
            if this is true then the parameters are passed directly to ABC
            rather than being drawn in the ABC. this is used by the PMC.
        bar : func
            the function for the progress bar. this must be different depending
            on whether this is run in a notebook or not.
        parameters : ndarray
            the parameter values to run the simulations at
        sims : ndarray
            the simulations to compress to perform the ABC
        estimates : ndarray
            the estimates of the simulations from the IMNN
        differences : ndarray
            the difference between the observed data and all of the estimates
        distances : ndarray
            the distance mesure describing how similar the observed estimate is
            to the estimates of the simulations
        """
        if predrawn:
            parameters = draws
            draws = parameters.shape[0]
        else:
            parameters = self.prior.sample(draws)
        if at_once:
            sims = self.simulator(parameters)
            if save_sims is not None:
                np.savez(save_sims + ".npz", sims)
            estimates = self.get_estimate(sims)
        else:
            estimates = np.zeros([draws, self.n_params])
            for theta in bar(range(draws), desc="Simulations"):
                sim = self.simulator([parameters[theta]])
                if save_sims is not None:
                    np.savez(save_sims + "_" + str(theta), sim)
                estimates[theta] = self.get_estimate([sim])[0]
        differences = estimates[:, np.newaxis, :] - self.estimate
        distances = np.sqrt(
            np.einsum(
                'ijk,ijk->ij',
                differences,
                np.einsum(
                    'ij,klj->kli',
                    self.F,
                    differences)))
        if update:
            parameters = np.concatenate(
                [self.parameters, parameters])
            estimates = np.concatenate(
                [self.estimates, estimates])
            differences = np.concatenate(
                [self.differences, differences])
            distances = np.concatenate(
                [self.distances, distances])
            self.parameters = parameters
            self.estimates = estimates
            self.differences = differences
            self.distances = distances
        return parameters, estimates, differences, distances

    def reset(self):
        self.parameters = np.array([]).reshape((0, self.n_params))
        self.differences = np.array([]).reshape((0, self.targets, self.n_params))
        self.estimates = np.array([]).reshape((0, self.n_params))
        self.distances = np.array([]).reshape((0, self.targets))
        self.num_accepted = None
        self.num_rejected = None
        self.num_draws = None
        self.accepted_parameters = None
        self.accepted_differences = None
        self.accepted_estimates = None
        self.accepted_distances = None
        self.rejected_parameters = None
        self.rejected_differences = None
        self.rejected_estimates = None
        self.rejected_distances = None
        self.grid = None
        self.post = None

    def accept_reject(self, ϵ):
        if self.parameters.shape[0] == 0:
            print("The ABC has not yet been called. Pass `draws` or run ABC(draws) "
                  "where draws is the desired number of simulations.")
            sys.exit()
        self.num_draws = self.distances.shape[0]
        accepted = np.array([
            np.argwhere(self.distances[:, i] < ϵ)[:, 0]
            for i in range(self.targets)])
        rejected = np.array([
            np.argwhere(self.distances[:, i] >= ϵ)[:, 0]
            for i in range(self.targets)])
        self.num_accepted = np.array([
            indices.shape[0] for indices in accepted])
        self.num_rejected = np.array([
            indices.shape[0] for indices in rejected])
        self.accepted_parameters = np.array([
            self.parameters[indices] for indices in accepted])
        self.rejected_parameters = np.array([
            self.parameters[indices] for indices in rejected])
        self.accepted_differences = np.array([
            self.differences[indices, i] for i, indices in enumerate(accepted)])
        self.rejected_differences = np.array([
            self.differences[indices, i] for i, indices in enumerate(rejected)])
        self.accepted_estimates = np.array([
            self.estimates[indices] for indices in accepted])
        self.rejected_estimates = np.array([
            self.estimates[indices] for indices in rejected])
        self.accepted_distances = np.array([
            self.distances[indices, i] for i, indices in enumerate(accepted)])
        self.rejected_distances = np.array([
            self.distances[indices, i] for i, indices in enumerate(rejected)])

    def get_min_accepted(self, ϵ, accepted, min_draws=1, at_once=True,
                         save_sims=None, tqdm_notebook=True):
        if min_draws is None:
            min_draws = 1
        if self.parameters.shape[0] == 0:
            self.__call__(draws=min_draws, at_once=at_once,
                          save_sims=save_sims)
        self.accept_reject(ϵ=ϵ)
        if np.any(self.num_accepted < accepted):
            if utils().isnotebook(tqdm_notebook):
                bar = tqdm.tqdm_notebook(total=np.inf, desc="Draws")
            else:
                bar = tqdm.tqdm(total=np.inf, desc="Draws")
        while np.any(self.num_accepted < accepted):
            self.__call__(draws=min_draws, at_once=at_once,
                          save_sims=save_sims)
            self.accept_reject(ϵ=ϵ)
            bar.update(self.num_draws)
            bar.set_postfix(Accepted=self.num_accepted, Remaining=accepted-self.num_accepted)

    def posterior(self, bins=25, ranges=None, **kwargs):
        self.setup_points(**kwargs)
        if ranges is None:
            low_ind = np.argwhere(self.prior.low == -np.inf)
            low = self.prior.low
            if len(low_ind) != 0:
                low[low_ind] = np.min(self.parameters, axis=(0, 1))[low_ind]
            high_ind = np.argwhere(self.prior.high == np.inf)
            high = self.prior.high
            if len(high_ind) != 0:
                high[high_ind] = np.max(self.parameters, axis=(0, 1))[high_ind]
            ranges = [(low[i], high[i]) for i in range(self.n_params)]
        temp = [np.histogramdd(parameters, density=True, range=ranges, bins=bins)
                for parameters in self.accepted_parameters]
        self.post = np.concatenate(
            [temp[i][0][np.newaxis, ...] for i in range(self.targets)],
            axis=0)
        self.grid = np.array([
            temp[0][1][i][:-1] + (temp[0][1][i][1] - temp[0][1][i][0]) / 2.
            for i in range(self.n_params)])
        return self.post

    def plot(self, smoothing=None, **kwargs):
        posterior = self.posterior(**kwargs)
        return self.triangle_plot(posterior, grid=self.grid, meshgrid=False,smoothing=smoothing, **kwargs)

    def setup_points(self, ϵ=None, draws=None, accepted=None, at_once=True, save_sims=None, tqdm_notebook=True, **kwargs):
        if ϵ is not None:
            if accepted is not None:
                self.get_min_accepted(ϵ=ϵ, accepted=accepted, min_draws=draws, at_once=at_once, save_sims=save_sims, tqdm_notebook=tqdm_notebook)
            elif draws is not None:
                self.__call__(draws=draws, at_once=at_once,
                              save_sims=save_sims)
                self.accept_reject(ϵ=ϵ)
            else:
                self.accept_reject(ϵ=ϵ)
        if self.accepted_parameters is None:
            print("The ABC acceptance and rejection has not yet been called. "
                  "Pass `ϵ` (and `draws` if the ABC has not been called).")
            sys.exit()

    def _scatter_plot(self, axes="parameter_estimate", rejected=0.1, ax=None, figsize=None, wspace=0, hspace=0, **kwargs):
        if rejected > 0:
            plot_rejected = True
        else:
            plot_rejected = False
        for i in range(self.targets):
            if self.rejected_parameters[i].shape[0] == 0:
                plot_rejected = False
        if self.targets > 1:
            accepted_labels = ["Accepted simulations {}".format(i+1)
                               for i in range(self.targets)]
            if plot_rejected:
                rejected_labels = ["Rejected simulations {}".format(i+1)
                                   for i in range(self.targets)]
        else:
            accepted_labels = ["Accepted simulations"]
            if plot_rejected:
                rejected_labels = ["Rejected simulations"]
        if axes == "parameter_estimate":
            x_accepted = self.accepted_parameters
            y_accepted = self.accepted_estimates
            if plot_rejected:
                x_rejected = np.array([self.rejected_parameters[i][::int(1/rejected)]
                                       for i in range(self.targets)])
                y_rejected = np.array([self.rejected_estimates[i][::int(1/rejected)]
                                       for i in range(self.targets)])
            axhline = self.estimate
            axvline = None
            if self.labels is not None:
                xlabels = self.labels
            else:
                xlabels = ["Parameter {}".format(i+1) for i in range(self.n_params)]
            ylabels = ["Estimate {}".format(i+1) for i in range(self.n_params)]
        elif axes == "parameter_parameter":
            x_accepted = self.accepted_parameters
            y_accepted = x_accepted
            if plot_rejected:
                x_rejected = np.array([self.rejected_parameters[i][::int(1/rejected)]
                                       for i in range(self.targets)])
                y_rejected = x_rejected
            axhline = None
            axvline = None
            if self.labels is not None:
                xlabels = self.labels
            else:
                xlabels = ["Parameter {}".format(i+1) for i in range(self.n_params)]
            ylabels = xlabels
        elif axes == "estimate_estimate":
            x_accepted = self.accepted_estimates
            y_accepted = x_accepted
            if plot_rejected:
                x_rejected = np.array([self.rejected_estimates[i][::int(1/rejected)]
                                       for i in range(self.targets)])
                y_rejected = x_rejected
            xlabels = ["Estimate {}".format(i+1) for i in range(self.n_params)]
            ylabels = xlabels
            axhline = self.estimate
            axvline = self.estimate
        else:
            print("`axes` must be `'parameter_estimate'`, `'parameter_parameter'` "
                  "or `'estimate_estimate'`")
            sys.exit()
        if ax is None:
            fig, ax = plt.subplots(self.n_params, self.n_params, figsize=figsize)
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        for i in range(self.n_params):
            for j in range(self.n_params):
                if j < self.n_params - 1:
                    ax[j, i].set_xticks([])
                if i > 0:
                    ax[j, i].set_yticks([])
                if i == 0:
                    ax[j, i].set_ylabel(ylabels[j])
                if j == self.n_params - 1:
                    ax[j, i].set_xlabel(xlabels[i])
                ax[0, i].get_shared_x_axes().join(
                    ax[0, i], ax[j, i])
                ax[j, 0].get_shared_y_axes().join(
                    ax[j, 0], ax[j, i])
                if plot_rejected:
                    for k in range(self.targets):
                        ax[j, i].scatter(x_rejected[k][:, i], y_rejected[k][:, j],
                                         s=1, label=rejected_labels[k])
                for k in range(self.targets):
                    ax[j, i].scatter(x_accepted[k][:, i], y_accepted[k][:, j],
                                     s=1, label=accepted_labels[k])
                if axhline is not None:
                    for k in range(self.targets):
                        ax[j, i].axhline(axhline[k, j], linestyle="dashed",
                                         color="black")
                if axvline is not None:
                    for k in range(self.targets):
                        ax[j, i].axvline(axvline[k, i], linestyle="dashed",
                                         color="black")
        ax[0, 0].legend(frameon=False,
                        bbox_to_anchor=(self.n_params+1, self.n_params-1))
        return ax

    def scatter_plot(self, ϵ=None, draws=None, accepted=None, at_once=True, save_sims=None, tqdm_notebook=True, **kwargs):
        self.setup_points(ϵ=ϵ, draws=draws, accepted=accepted, at_once=at_once, save_sims=save_sims, tqdm_notebook=tqdm_notebook)
        self._scatter_plot(**kwargs)

class PopulationMonteCarlo(ApproximateBayesianComputation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, draws, initial_draws, criterion, percentile=75,
                 at_once=True, save_sims=None, tqdm_notebook=True):
        self.PMC(draws=draws, initial_draws=initial_draws, criterion=criterion,
                 percentile=percentile, at_once=at_once, save_sims=save_sims, tqdm_notebook=tqdm_notebook)

    def PMC(self, draws, initial_draws, criterion, percentile=75,
            at_once=True, save_sims=None, tqdm_notebook=True):
        """Population Monte Carlo

        This is the population Monte Carlo sequential ABC method, highly
        optimised for minimal numbers of drawsself.

        It works by first running an ABC and sorting the output distances,
        keeping the closest n parameters (where n is the number of samples to
        keep for the posterior) to get an initial proposal distribution.

        The proposal distribution is a Gaussian distribution with covariance
        given by weighted parameter values. Each iteration of draws moves 25%
        of the futhers samples until they are within the epsilon for that
        iteration. Once this is done the new weighting is calculated depending
        on the value of the new parameters and the new weighted covariance is
        calculated.

        Convergence is classified with a criterion which compares how many
        draws from the proposal distribution are needed to be accepted. When
        the number of draws is large then the posterior is fairly stable and
        can be trusted to really be approaching the true posterior distribution

        Parameters
        __________
        draws : int
            number of parameter draws from the prior to make on initialisation
        posterior : int
            number of samples to keep from the final provided posterior
        criterion : float
            the ratio to the number of samples to obtain in from the final
            posterior to the number of draws needed in an iteration.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        restart : bool, optional
            to restart the PMC from scratch set this value to true, otherwise
            the PMC just carries on from where it last left off. note that the
            weighting is reset, but it should level itself out after the first
            iteration.
        iteration : int
            counter for the number of iterations of the PMC to convergence.
        criterion_reached : float
            the value of the criterion after each iteration. once this reaches
            the supplied criterion value then the PMC stops.
        weighting : ndarray
            the weighting of the covariance for the proposal distribution.
        cov : ndarray
                the covariance of the parameter samples for the proposal
            distribution.
        epsilon : float
            the distance from the summary of the observed data where the
            samples are accepted.
        stored_move_ind : list
            the indices of the most distant parameter values which need to be
            moved during the PMC.
        move_ind : list
            the indices of the stored_move_ind which is decreased in size until
            all of the samples have been moved inside the epsilon.
        current_draws : int
            the number of draws taken when moving the samples in the population
        accepted_parameters : ndarray
            the parameter values which have been successfully moved during PMC.
        accepted_estimates : ndarray
            the estimates which have successfully moved closer than epsilon.
        accepted_differences : ndarray
            the difference between the observed data and all of the summaries.
        accepted_distances : ndarray
            the distance mesure describing how similar the observed summary is
            to the summaries of the simulations.
        proposed_parameters : ndarray
            the proposed parameter values to run simulations at to try and move
            closer to the true observation.
        temp_dictionary : dict
            dictionary output of the ABC with all summaries, parameters and
            distances.
        accept_index : list
            the indices of the accepted samples.
        reject_index : list
            the indices of the rejected samples.
        inv_cov : ndarray
            inverse covariance for the Gaussian proposal distribution.
        dist : ndarray
            the value of the proposal distribution at the accepted parameters.
        diff : ndarray
            difference between the accepted parameters and the parameter values
            from the previous iteration.
        """
        if ((draws is None) or (criterion is None) or (initial_draws is None)) and (self.parameters.shape[0] == 0):
            print("PMC has not yet been run. Please pass `draws`, `initial_draws` and a criterion value.")
            sys.exit()
        if draws is not None:
            if initial_draws < draws:
                print("`initial_draws` must be equal to or greater than `draws`.")
                sys.exit()
            if draws > self.parameters.shape[0]:
                if self.parameters.shape[0] == 0:
                    self.ABC(initial_draws, at_once=at_once, save_sims=save_sims)
                    self.parameters = np.repeat(
                        self.parameters[np.newaxis, ...],
                        self.targets, axis=0)
                    self.estimates = np.repeat(
                        self.estimates[np.newaxis, ...],
                        self.targets, axis=0)
                    self.differences = np.moveaxis(self.differences, 0, 1)
                    self.distances = np.moveaxis(self.distances, 0, 1)
                    self.weighting = np.ones((self.targets, initial_draws)) / draws
                    self.num_draws = np.zeros(self.targets)
                else:
                    parameters, estimates, differences, distances = self.ABC(initial_draws, at_once=at_once, save_sims=save_sims, update=False)
                    self.parameters = np.concatenate(
                        [self.parameters,
                         np.repeat(
                             parameters[np.newaxis, ...],
                             self.targets, axis=0)], axis=1)
                    self.estimates = np.concatenate(
                        [self.estimates,
                         np.repeat(
                             estimates[np.newaxis, ...],
                             self.targets, axis=0)], axis=1)
                    self.differences = np.concatenate(
                        [self.differences,
                         np.moveaxis(differences, 0, 1)], axis=1)
                    self.distances = np.concatenate(
                        [self.distances,
                         np.moveaxis(distances, 0, 1)], axis=1)
                    self.weighting = np.concatenate(
                        [self.weighting, np.zeros((self.targets, initial_draws))], axis=1)
                self.sort(draws=draws)
            if percentile is None:
                ϵ_ind = -1
                to_accept = 1
            else:
                ϵ_ind = int(percentile / 100 * draws)
                to_accept = draws - ϵ_ind
            iteration = 0
            criterion_reached = np.greater(np.ones(self.targets) * np.inf, criterion)
            if utils().isnotebook(tqdm_notebook):
                bar = tqdm.tqdm_notebook(total=np.inf, desc="Iterations")
            else:
                bar = tqdm.tqdm(total=np.inf, desc="Iterations")
            while np.any(criterion_reached):
                targets = np.argwhere(criterion_reached)[:, 0]
                iteration_draws = np.zeros(targets.shape[0])
                cov = np.array([
                    np.cov(
                        self.parameters[i],
                        aweights=self.weighting[i],
                        rowvar=False)
                    for i in targets])
                if self.n_params == 1:
                    cov = cov[:, np.newaxis, np.newaxis]
                inv_cov = np.linalg.inv(cov)
                ϵ = self.distances[targets, ϵ_ind]
                a_ind = np.arange(to_accept * targets.shape[0])
                t_ind = np.repeat(np.arange(targets.shape[0]), to_accept)
                params = self.estimates[targets, ϵ_ind:].reshape(
                    (-1, self.n_params))
                ests = self.estimates[targets, ϵ_ind:].reshape(
                    (-1, self.n_params))
                dist = self.distances[targets,  ϵ_ind:].reshape(-1)
                diff = self.differences[targets,  ϵ_ind:].reshape(
                    (-1, self.n_params))
                loc = self.parameters[targets, ϵ_ind:].reshape(
                    (-1, self.n_params))
                scale = np.repeat(
                    np.linalg.cholesky(cov),
                    to_accept,
                    axis=0)
                while a_ind.shape[0] > 0:
                    samples = np.zeros((a_ind.shape[0], self.n_params))
                    s_ind = np.arange(a_ind.shape[0])
                    while s_ind.shape[0] > 0:
                        u = np.random.normal(0, 1, loc[a_ind[s_ind]].shape)
                        samples[s_ind] = loc[a_ind[s_ind]] + np.einsum("ijk,ik->ij", scale[a_ind[s_ind]], u)
                        accepted = np.logical_and(
                            np.all(
                                np.greater(
                                    samples[s_ind],
                                    self.prior.low),
                                axis=1),
                            np.all(
                                np.less(
                                    samples[s_ind],
                                    self.prior.high),
                                axis=1))
                        s_ind = s_ind[~accepted]
                    parameters, estimates, differences, distances = self.ABC(
                        samples, at_once=at_once, save_sims=save_sims, predrawn=True, update=False)
                    distances = np.diag(distances[:, t_ind])
                    differences = np.vstack([np.diag(differences[:, t_ind, i]) for i in range(self.n_params)]).T
                    accepted = np.less(
                        distances,
                        np.take(ϵ, t_ind))
                    dist[a_ind[accepted]] = distances[accepted]
                    diff[a_ind[accepted]] = differences[accepted]
                    ests[a_ind[accepted]] = estimates[accepted]
                    params[a_ind[accepted]] = parameters[accepted]
                    iteration_draws = np.array([iteration_draws[i] + np.sum(t_ind == i) for i in range(targets.shape[0])])
                    a_ind = a_ind[~accepted]
                    t_ind = t_ind[~accepted]
                this_weighting_norm = lambda x : self.weighting_norm(x, self.parameters[targets], inv_cov, self.weighting[targets])
                self.parameters[targets, ϵ_ind:] = params.reshape(
                    (targets.shape[0], to_accept, self.n_params))
                self.estimates[targets, ϵ_ind:] = ests.reshape((targets.shape[0], to_accept, self.n_params))
                self.distances[targets, ϵ_ind:] = dist.reshape((targets.shape[0], to_accept))
                self.differences[targets, ϵ_ind:] = diff.reshape((targets.shape[0], to_accept, self.n_params))
                self.weighting[targets] = self.prior.prob(self.parameters[targets]).numpy() / this_weighting_norm(self.parameters[targets])
                self.sort(index=targets)
                this_criterion = draws / iteration_draws
                criterion_reached = np.greater(this_criterion, criterion)
                iteration += 1
                self.num_draws[targets] += iteration_draws
                bar.update(iteration)
                bar.set_postfix(criterion=this_criterion, draws=self.num_draws, ϵ=ϵ)

    def weighting_norm(self, parameters, means, inverse_covariance, weighting):
        diff = parameters[:, np.newaxis, ...] - means[:, :, np.newaxis, ...]
        exp = -0.5 * np.einsum("ijkl,ilm,ijkm->ijk", diff, inverse_covariance, diff)
        norm = -0.5 * np.log(2. * np.pi * np.linalg.det(inverse_covariance))[:, np.newaxis, np.newaxis]
        return np.sum(np.exp(exp + norm) * weighting[:, :, np.newaxis], 1)

    def sort(self, index=None, draws=None):
        indices = self.distances.argsort(axis=1)
        if draws is not None:
            indices = indices[:, :draws]
        if index is None:
            self.parameters = np.array([self.parameters[i, ind] for i, ind in enumerate(indices)])
            self.estimates = np.array([self.estimates[i, ind] for i, ind in enumerate(indices)])
            self.differences = np.array([self.differences[i, ind] for i, ind in enumerate(indices)])
            self.distances = np.array([self.distances[i, ind] for i, ind in enumerate(indices)])
            self.weighting = np.array([self.weighting[i, ind] for i, ind in enumerate(indices)])
        else:
            self.parameters[index] = np.array([self.parameters[i, indices[i]] for i in index])
            self.estimates[index] = np.array([self.estimates[i, indices[i]] for i in index])
            self.differences[index] = np.array([self.differences[i, indices[i]] for i in index])
            self.distances[index] = np.array([self.distances[i, indices[i]] for i in index])
            self.weighting[index] = np.array([self.weighting[i, indices[i]] for i in index])

    def setup_points(self, draws=None, initial_draws=None, criterion=None, percentile=75, at_once=True, save_sims=None, **kwargs):
        self.__call__(draws=draws, initial_draws=initial_draws, criterion=criterion, percentile=percentile, at_once=at_once, save_sims=save_sims)
        self.accepted_parameters = self.parameters
        self.accepted_estimates = self.estimates
        self.rejected_parameters = np.array([]).reshape((self.targets, 0))
        self.rejected_estimates = np.array([]).reshape((self.targets, 0))

    def scatter_plot(self, draws=None, initial_draws=None, criterion=None, percentile=None, at_once=True, save_sims=None, **kwargs):
        self.setup_points(draws=draws, initial_draws=initial_draws, criterion=criterion, percentile=percentile, at_once=at_once, save_sims=save_sims)
        self._scatter_plot(**kwargs)

class GaussianApproximation(LFI):
    def __init__(self, **kwargs):
        setattr(self, "log_like", None)
        setattr(self, "log_post", None)
        setattr(self, "log_prior", None)
        setattr(self, "grid", None)
        setattr(self, "shape", None)
        super().__init__(simulator=None, **kwargs)

    def __call__(self, grid=None, gridsize=20, prior=True):
        self.check_prerun(grid, gridsize, prior)

    def log_gaussian(self, grid, shape):
        diff = self.estimate[:, np.newaxis, :] - grid[np.newaxis, ...]
        exp = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, self.F, diff)
        norm = -0.5 * np.log(2. * np.pi * np.linalg.det(self.Finv))
        return np.reshape(exp + norm,((-1,) + shape))

    def calculate_likelihood(self, grid, shape, prior):
        self.log_like = self.log_gaussian(grid, shape)
        if prior:
            self.log_prior = np.reshape(
                self.prior.log_prob(grid),
                ((-1,) + shape))
            self.log_post = self.log_like + self.log_prior
        self.grid = np.reshape(grid.T, (-1,) + shape)

    def check_prerun(self, grid, gridsize, prior):
        if (self.log_like is not None):
            if grid is not None:
                grid, shape = self.check_grid(grid)
                if not np.all(self.grid == grid):
                    self.calculate_likelihood(grid, shape, prior)
            else:
                grid, shape = self.make_grid(gridsize)
                if not np.all(self.grid == grid):
                    self.calculate_likelihood(grid, shape, prior)
        else:
            if grid is not None:
                grid, shape = self.check_grid(grid)
                self.calculate_likelihood(grid, shape, prior)
            else:
                grid, shape = self.make_grid(gridsize)
                self.calculate_likelihood(grid, shape, prior)

    def check_grid(self, grid):
        if len(grid.shape) == 1:
            this_grid = grid[np.newaxis, :]
        elif len(grid.shape) == 2:
            this_grid = grid.T
        else:
            this_grid = grid.reshape((self.n_params, -1)).T
        return this_grid, grid[0].shape

    def make_grid(self, gridsize):
        gridsize = self.utils.check_gridsize(gridsize, self.n_params)
        parameters = [np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                gridsize[i])
            for i in range(self.n_params)]
        return self.check_grid(
            np.array(
                np.meshgrid(*parameters, indexing="ij")))

    def log_prob(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize, prior=False)
        return self.log_like

    def prob(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize, prior=False)
        return np.exp(self.log_like)

    def log_posterior(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize)
        return self.log_post

    def posterior(self, grid=None, gridsize=20):
        self.__call__(grid=grid, gridsize=gridsize)
        return np.exp(self.log_post)

    def plot(self, grid=None, gridsize=20, **kwargs):
        posterior = self.posterior(grid=grid, gridsize=gridsize)
        return self.triangle_plot(posterior, self.grid, **kwargs)
