import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from imnn_tf.utils import TFRecords
from imnn_tf.lfi import GaussianApproximation

__version__ = "0.2.0"
__author__ = "Tom Charnock"

class GenerateGaussianNoise():
    def __init__(self, input_shape=(10,), n_params=2, n_summaries=2, n_s=1000, n_d=1000, n_d_small=100,
                 θ_fid=np.array([0., 1.]), δθ=np.array([0.1, 0.1]), training_seed=0,
                 validation_seed=1):
        self.input_shape = input_shape
        self.n_params = n_params
        self.n_summaries = n_summaries
        self.n_s = n_s
        self.n_d = n_d
        self.n_d_small = n_d_small
        self.θ_fid = θ_fid
        self.δθ = δθ
        self.half_δθ = δθ / 2.
        self.training_seed = training_seed
        self.validation_seed = validation_seed

    def get_fiducial(self, seed, data):
        return data[seed]

    def get_derivative(self, seed, derivative, parameter, data):
        return data[seed, derivative, parameter]

    def check_selection(self, size):
        if size not in ["full", "all", "small"]:
            print("size must be `full`, `all` or `small` describing, respectively "
                  "whether just `n_d=n_s` is returned, or `n_d=n_s` and `n_d_small` "
                  "is returned, or `n_d=n_d_small` is returned.")
            sys.exit()

    def check_ftype(self, ftype):
        if ftype not in ["both", "numpy", "tfrecords"]:
            print("size must be `both`, `numpy` or `tfrecords` describing, respectively "
                  "whether both `numpy` and `tfrecords` files are saved, or just either one.")
            sys.exit()

    def simulator(self, parameters, seed=None, simulator_args=None):
        if seed is not None:
            np.random.seed(seed)
        if len(parameters.shape) == 1:
            parameters = parameters[np.newaxis, :]
        if self.n_params == 1:
            parameters = np.repeat(parameters, 2, axis=1)
            parameters[:, 0] = np.zeros_like(parameters[:, 0])
        return np.moveaxis(
            np.random.normal(
                parameters[:, 0],
                np.sqrt(parameters[:, 1]),
                self.input_shape + (parameters.shape[0],)),
            -1, 0)

    def generate_data(self, size="full"):
        self.check_selection(size)
        details = dict(
            input_shape=self.input_shape,
            n_params=self.n_params,
            n_summaries=self.n_summaries,
            n_s=self.n_s,
            n_d=self.n_d,
            θ_fid=self.θ_fid,
            δθ=self.δθ)

        a_0 = self.simulator(
            parameters=np.repeat(
                self.θ_fid[np.newaxis, :],
                self.n_s,
                axis=0),
            seed=self.training_seed,
            simulator_args={"input_shape": self.input_shape})
        a_1 = self.simulator(
            parameters=np.repeat(
                self.θ_fid[np.newaxis, :],
                self.n_s,
                axis=0),
            seed=self.validation_seed,
            simulator_args={"input_shape": self.input_shape})

        b_0 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0] - self.half_δθ[0],
                    self.θ_fid[1]])[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.training_seed,
            simulator_args={"input_shape": self.input_shape})
        b_1 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0] - self.half_δθ[0],
                    self.θ_fid[1]])[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.validation_seed,
            simulator_args={"input_shape": self.input_shape})
        c_0 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0] + self.half_δθ[0],
                    self.θ_fid[1]])[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.training_seed,
            simulator_args={"input_shape": self.input_shape})
        c_1 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0] + self.half_δθ[0],
                    self.θ_fid[1]])[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.validation_seed,
            simulator_args={"input_shape": self.input_shape})
        d_0 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0],
                    self.θ_fid[1] - self.half_δθ[1]]
                )[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.training_seed,
            simulator_args={"input_shape": self.input_shape})
        d_1 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0],
                    self.θ_fid[1] - self.half_δθ[1]]
                )[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.validation_seed,
            simulator_args={"input_shape": self.input_shape})
        e_0 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0],
                    self.θ_fid[1] + self.half_δθ[1]]
                )[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.training_seed,
            simulator_args={"input_shape": self.input_shape})
        e_1 = self.simulator(
            parameters=np.repeat(
                np.array([
                    self.θ_fid[0],
                    self.θ_fid[1] + self.half_δθ[1]]
                )[np.newaxis, :],
                self.n_d,
                axis=0),
            seed=self.validation_seed,
            simulator_args={"input_shape": self.input_shape})

        f_0 = np.stack((np.stack((b_0, c_0)),
                        np.stack((d_0, e_0)))
                      ).transpose(2, 1, 0, 3)
        f_1 = np.stack((np.stack((b_1, c_1)),
                        np.stack((d_1, e_1)))
                      ).transpose(2, 1, 0, 3)

        result = (details, a_0, a_1, f_0, f_1)

        #TO ADD
        #if analytic_derivative:
        #    f_0 = (f_0[:, 1] - f_0[:, 0]) / self.δθ[np.newaxis, ...]
        #    f_1 = (f_1[:, 1] - f_1[:, 0]) / self.δθ[np.newaxis, ...]
        if size == "all":
            details["n_d_small"] = self.n_d_small
            result += (f_0[:self.n_d_small],
                        f_1[:self.n_d_small])
        elif size == "small":
            details["n_d"] = self.n_d_small
            result[-2] = f_0[:self.n_d_small]
            result[-1] = f_1[:self.n_d_small]

        return result

    def save(self, ftype="both", size="full", directory="data", record_size=0.01):
        self.check_ftype(ftype)
        result = self.generate_data(size=size)

        if (ftype=="both") or (ftype=="numpy"):
            np.savez("{}/details.npz".format(directory), result[0])
            np.save("{}/fiducial.npy".format(directory), result[1])
            np.save("{}/validation_fiducial.npy".format(directory), result[2])
            np.save("{}/derivative.npy".format(directory), result[3])
            np.save("{}/validation_derivative.npy".format(directory), result[4])
            if size == "all":
                np.save("{}/derivative_small.npy".format(directory), result[5])
                np.save("{}/validation_derivative_small.npy".format(directory), result[6])

        if (ftype=="both") or (ftype=="tfrecords"):
            writer = TFRecords(record_size=record_size)

            writer.write_record(
                n_sims=result[0]["n_s"],
                get_simulation=lambda x : self.get_fiducial(x, result[1]),
                fiducial=True,
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_s"],
                get_simulation=lambda x : self.get_fiducial(x, result[2]),
                fiducial=True,
                validation=True,
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"],
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[3]),
                fiducial=False,
                n_params=result[0]["n_params"],
                directory="{}/tfrecords".format(directory))
            writer.write_record(
                n_sims=result[0]["n_d"],
                get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[4]),
                fiducial=False,
                n_params=result[0]["n_params"],
                validation=True,
                directory="{}/tfrecords".format(directory))
            if size == "all":
                writer.write_record(
                    n_sims=result[0]["n_d_small"],
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[5]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")
                writer.write_record(
                    n_sims=result[0]["n_d_small"],
                    get_simulation=lambda x, y, z : self.get_derivative(x, y, z, result[6]),
                    fiducial=False,
                    n_params=result[0]["n_params"],
                    directory="{}/tfrecords".format(directory),
                    filename="derivative_small")

    def plot_data(self, data, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = (5, 4))
        ax.plot(data.T, label=label)
        ax.legend(frameon=False)
        ax.set_xlim([0, data.shape[-1] - 1])
        ax.set_xticks([])
        ax.set_ylabel("Data amplitude");

class AnalyticLikelihood(GaussianApproximation):
    def __init__(self, data, prior, generator, parameters=2, labels=None):
        if parameters == 2:
            self.log_gaussian = self._mean_variance_likelihood
            self.Fisher = self._mean_variance_Fisher
            self.get_estimate = self._get_mean_variance
        elif parameters == 1:
            self.log_gaussian = self._variance_likelihood
            self.Fisher = self._variance_Fisher
            self.get_estimate = self._get_variance
        else:
            print("`parameters` must be 2 for mean and variance or 1 for just variance")
            sys.exit()
        super().__init__(
            target_data=data,
            prior=prior,
            Fisher=None,
            get_estimate=self.get_estimate,
            labels=labels)

    def _mean_variance_likelihood(self, grid, shape):
        sq_diff = (self.data[..., np.newaxis] - grid[:, 0])**2.
        exp = np.sum(-0.5 * sq_diff / grid[:, 1], axis=1)
        norm = -(self.data.shape[1] / 2.) * np.log(
            2. * np.pi * grid[:, 1])[np.newaxis, ...]
        return np.reshape(exp + norm, ((-1,) + shape))

    def _variance_likelihood(self, grid, shape):
        sq = self.data[..., np.newaxis]**2.
        exp = np.sum(-0.5 * sq / grid[:, 0], axis=1)
        norm = -(self.data.shape[1] / 2.) * np.log(
            2. * np.pi * grid[:, 0])[np.newaxis, ...]
        return np.reshape(exp + norm, ((-1,) + shape))

    def _mean_variance_Fisher(self, θ_fid):
        return -np.array([
            [- np.prod(self.data.shape[1:]) / θ_fid[1], 0.],
            [0. , - 0.5 * np.prod(self.data.shape[1:]) / θ_fid[1]**2.]])

    def _variance_Fisher(self, θ_fid):
        return -np.array([[- 0.5 * np.prod(self.data.shape[1:]) / θ_fid[0]**2.]])

    def _get_mean_variance(self, data):
        return np.array([np.mean(data, axis=1),
                         np.std(data, axis=1)**2]).T

    def _get_variance(self, data):
        return np.array([np.std(data, axis=1)**2]).T


def main(args):
    data = GenerateGaussianNoise(
        input_shape=tuple(args.input_shape),
        n_params=args.n_params,
        n_summaries=args.n_summaries,
        n_s=args.n_s,
        n_d=args.n_d,
        n_d_small=args.n_d_small,
        fiducial=np.array(args.fiducial),
        delta=np.array(args.delta),
        training_seed=args.training_seed,
        validation_seed=args.validation_seed)
    data.save(
        ftype=args.ftype,
        size=args.size,
        directory=args.directory,
        record_size=args.record_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', nargs='*', default=[10], type=int)
    parser.add_argument('--n_params', nargs='?', const=1, default=2, type=int)
    parser.add_argument('--n_summaries', nargs='?', const=1, default=2, type=int)
    parser.add_argument('--n_s', nargs='?', const=1, default=1000, type=int)
    parser.add_argument('--n_d', nargs='?', const=1, default=1000, type=int)
    parser.add_argument('--n_d_small', nargs='?', const=1, default=100, type=int)
    parser.add_argument('--fiducial', nargs='*', default=[0, 1], type=float)
    parser.add_argument('--delta', nargs='*', default=[0.1, 0.1], type=float)
    parser.add_argument('--training_seed', nargs='?', const=1, default=0, type=int)
    parser.add_argument('--validation_seed', nargs='?', const=1, default=1, type=int)
    parser.add_argument('--ftype', nargs='?', const=1, default="both", type=str)
    parser.add_argument('--size', nargs='?', const=1, default="all", type=str)
    parser.add_argument('--directory', nargs='?', const=1, default="data", type=str)
    parser.add_argument('--record_size', nargs='?', const=1, default=0.01, type=float)
    args = parser.parse_args()
    main(args)
