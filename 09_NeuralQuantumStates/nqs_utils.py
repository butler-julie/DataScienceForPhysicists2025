import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
jax.config.update("jax_enable_x64", True)
import optax


class NeuralQuantumState(object):
    """Constructs the internal neural network and defines the variational wave function."""

    def __init__(self,
        layers : list[int],
        activation : callable,
        conf: float
    ):

        self.layers = layers            # dimensions of each layer, including input and output
        self.activation = activation    # nonlinear activation function
        self.c = conf                   # confinement hyperparameter


    def build(self, key):
        """Returns the weights and biases of the internal neural network"""

        params = []

        for l in range(1, len(self.layers)):

            # define the input and output dimensions of layer l
            dim_in = self.layers[l-1]
            dim_out = self.layers[l]

            # define Gaussian widths of random initialization
            sigma_W = jnp.sqrt(2 / (dim_in + dim_out))
            sigma_b = 0.001

            # randomly initialize the weights and biases of layer l
            key, subkey_W, subkey_b = jax.random.split(key, num=3)
            W = sigma_W * jax.random.normal(subkey_W, (dim_in, dim_out))
            b = sigma_b * jax.random.normal(subkey_b, (dim_out,))

            # add the weights and biases of layer l to the parameters of the network
            params.append((W, b))

        # flatten parameters
        flat_params = self.flatten_params(params)
        num_params = flat_params.shape[0]
        print("Number of trainable parameters = ", num_params)

        return params



    @partial(jax.jit, static_argnames=('self'))
    def apply_net(self, params, x):
        """ Passes a single value x through the network. Possible to use in batches. """

        # apply hidden layers
        for W, b in params[:-1]:
            x = self.activation(jnp.dot(x, W) + b)

        # apply output layer
        W, b = params[-1]
        return jnp.dot(x, W) + b



    @partial(jax.jit, static_argnames=('self'))
    def logpsi(self, params, x):
        """ Defines the logarithm of the wave function in terms of the neural network. """
        F = self.apply_net(params, x)[0,0]    # get the output of the neural network
        return F - self.c * jnp.sum(x**2)     # apply constraint on boundary conditions



    @partial(jax.jit, static_argnames=('self'))
    def vmap_logpsi(self, params, x):
        """ Helper function for computing logpsi in batches. """
        vmap_logpsi = jax.vmap(self.logpsi, in_axes=(None, 0))(params, x)
        return vmap_logpsi



    @partial(jax.jit, static_argnames=('self'))
    def unflatten_params(self, flat_params):
        """ Helper function for constructing a pytree of parameters from a flat parameter vector. """
        params = self.unravel(flat_params)
        return params



    @partial(jax.jit, static_argnames=('self'))
    def flatten_params(self, params):
        """ Helper function for flattening a pytree into a parameter vector. """
        flat_params, self.unravel = ravel_pytree(params)
        return flat_params





class Sampler(object):
    """ Samples configurations from an instance of the `NeuralQuantumState` class. """

    def __init__(self,
        nchains : int,               # number of independent MC chains
        nsamples_per_chain : int,    # number of samples generated per chain
        neq : int,                   # number of equilibration steps before collecting samples
        nskip : int,                 # number of samples to skip between each collected sample
        sigma_step : float,          # width of Gaussian step
        sigma_init : float,          # width of initial Gaussian positions
        wavefunction : callable      # instance of NeuralQuantumState class
    ):

        self.nchains = nchains
        self.nsamples_per_chain = nsamples_per_chain
        self.neq = neq
        self.nskip = nskip
        self.sigma_step = sigma_step
        self.sigma_init = sigma_init
        self.wavefunction = wavefunction



    # performs one metropolis step for all chains
    @partial(jax.jit, static_argnames=('self'))
    def step(self, iter, vals):

        # unpack values
        key, params, x_o, logpsi_o = vals

        # propose random moves
        key, subkey = jax.random.split(key)
        x_n = x_o + self.sigma_step * jax.random.normal(subkey, (self.nchains,))

        # compute acceptance probability
        logpsi_n = self.wavefunction.vmap_logpsi(params, x_n)
        accept_prob = jnp.exp( 2 * (logpsi_n - logpsi_o) )

        # accept or reject proposal for each chain
        key, subkey = jax.random.split(key)
        unif = jax.random.uniform(subkey, (self.nchains,))
        accept = unif < accept_prob
        x_o = jnp.where(accept, x_n, x_o)
        logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)

        # repack and return values
        return (key, params, x_o, logpsi_o)


    @partial(jax.jit, static_argnames=('self'))
    def skip(self, iter, vals):
        """ Runs nskip metropolis steps for all chains. """

        key, params, x_o, logpsi_o = vals
        vals = jax.lax.fori_loop(0, self.nskip, self.step, vals)
        key, params, x_o, logpsi_o = vals

        return vals


    @partial(jax.jit, static_argnames=('self'))
    def sample(self, key, params):
        """ Returns equilbrated samples of positions. """

        # initialize all MC chains
        key, subkey = jax.random.split(key)
        x_o = self.sigma_init * jax.random.normal(subkey, (self.nchains,))

        # compute the wave function at the initial points
        logpsi_o = self.wavefunction.vmap_logpsi(params, x_o)

        # pack the values to loop over
        vals = (key, params, x_o, logpsi_o)

        # equilibrate sampler
        # loop over neq samples, skipping nskip samples between each one
        vals = jax.lax.fori_loop(0, self.neq, self.skip, vals)

        # store the remaining samples
        x_stored = jnp.zeros((self.nsamples_per_chain, self.nchains))
        for i in range(self.nsamples_per_chain):

            key, params, x_o, logpsi_o = self.skip(i, vals)
            x_stored = x_stored.at[i,:].set(x_o)
            vals = (key, params, x_o, logpsi_o)

        return jnp.reshape(x_stored, (-1))




class Hamiltonian(object):
    """The Hamiltonian acts as both input to the VMC method and the cost function. """

    def __init__(self,
    
        wavefunction : callable,
        potential : callable
    ):

        self.wavefunction = wavefunction   # An instance of the NeuralQuantumState class
        self.potential = potential         # A function defining the potential trap


    @partial(jax.jit, static_argnames=('self'))
    def kinetic(self, params, x):
        """ Computes the local kinetic energy by applying jax.grad twice. """
        dx_logpsi = lambda x: jax.grad(self.wavefunction.logpsi, argnums=1)(params, x)
        d2x_logpsi = jax.grad(dx_logpsi)
        return - 0.5 * ( d2x_logpsi(x) + dx_logpsi(x)**2  )

    @partial(jax.jit, static_argnames=('self'))
    def energy(self, params, x):
        """ Computes the local energies. """
        T = jax.vmap(self.kinetic, in_axes=(None, 0))(params, x).squeeze()
        V = self.potential(x).squeeze()
        return T + V


    @partial(jax.jit, static_argnames=('self'))
    def stats(self, energies):
        """ Computes statistical averages and errors. """
        nsamples = energies.shape[0]
        mean = jnp.mean(energies)
        mean2 = jnp.mean(energies**2)
        error = jnp.sqrt(mean2 - mean**2) / jnp.sqrt(nsamples-1)
        return mean, error

    @partial(jax.jit, static_argnames=('self'))
    def grad_params(self, params, x):
        """ Computes the gradient of logpsi with respect to params using jax.grad. """
        dp_logpsi = jax.grad(self.wavefunction.logpsi, argnums=0)(params, x)
        dp_logpsi = self.wavefunction.flatten_params(dp_logpsi)
        return dp_logpsi


    @partial(jax.jit, static_argnames=('self'))
    def vmap_grad_params(self, params, x):
        """ Helper function for computing batches of gradients. """
        return jax.vmap(self.grad_params, in_axes=(None, 0))(params, x)

    @partial(jax.jit, static_argnames=('self'))
    def grad_energy(self, params, x, energies):
        """ Computes the gradient of the energy with respect to params. """
        nsamples = energies.shape[0]
        dp_logpsi = self.vmap_grad_params(params, x)
        dp_logpsi = dp_logpsi - jnp.mean(dp_logpsi, axis=0)
        energies = energies - jnp.mean(energies)
        dp_energy = 2 * jnp.matmul(energies, dp_logpsi) / nsamples
        return dp_energy




class Optimizer(object):

    """ Uses Adam to train the NeuralQuanumState according to the given Hamiltonian. """

    def __init__(self,
    
        learning_rate : float,
        wavefunction : callable,
        sampler : callable,
        hamiltonian: callable
    ):
    
        self.wavefunction = wavefunction
        self.sampler = sampler
        self.hamiltonian = hamiltonian
        self.optimizer = optax.adam(learning_rate)


    @partial(jax.jit, static_argnames=('self'))
    def step(self, key, params, opt_state):
        """ A single optimization step. """

        # sample from the wave function
        key, subkey = jax.random.split(key)
        x = self.sampler.sample(subkey, params)

        # compute local energies for each sample
        E = self.hamiltonian.energy(params, x)

        # accumulate statistics
        E_avg, E_err = self.hamiltonian.stats(E)

        # compute gradient of energy expectation
        E_grad = self.hamiltonian.grad_energy(params, x, E)
        E_grad = self.wavefunction.unflatten_params( E_grad )

        updates, opt_state = self.optimizer.update(E_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return key, params, opt_state, E_avg, E_err, x


    def train(self, nopt, nbins, key, params):
        """ Loops over optimization steps. """

        # initialize optimizer
        opt_state = self.optimizer.init(params)
        
        # empty containers to track training
        energies = jnp.zeros(nopt,)
        errors = jnp.zeros(nopt,)
        histograms = jnp.zeros((nopt, nbins))
        bins = jnp.zeros((nopt, nbins))

        # heading
        print(f"\n{'iter':>5s} {'avg E':>10s} {'err E':>10s} {'time':>10s}")

        for iter in range(nopt):

            # update parameters
            ti = time.time()
            key, params, opt_state, E_avg, E_err, x = self.step(key, params, opt_state)
            tf = time.time()

            if iter % 10 == 0:
                print(f"{iter:5d} {E_avg:10.3e} {E_err:10.3e} {tf-ti:10.3e}")

            # store training
            energies = energies.at[iter].set(E_avg)
            errors = errors.at[iter].set(E_err)

            histogram, bin_edge = jnp.histogram(x, bins=nbins, density=True)
            histograms = histograms.at[iter,:].set(histogram)
            bins = bins.at[iter,:].set(0.5 * (bin_edge[1:] + bin_edge[:-1]))

        return params, energies, errors, histograms, bins



def animate(training, potential, xmin=-3, xmax=3, ymax=0.9, skip=2, E_exact=None):
    """ Animates the evolution of the wave function and energy using the output of Optimizer.train. """

    params, energies, errors, histograms, bins = training
    nopt = energies.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    def drawframe(n):
        
        N = n * skip
        
        ax[0].cla()
        ax[1].cla() 
        ax[0].grid(alpha=0.5)
        ax[1].grid(alpha=0.5)
        ax[0].set_xlim((xmin, xmax))
        ax[0].set_ylim((0, ymax))
        ax[1].set_xlim((0, N+1))
        ax[0].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$|\Psi(x)|^2$")
        ax[1].set_xlabel("Training Iteration")
        ax[1].set_ylabel("Energy")

        iters = jnp.arange(N)
        psi2 = ax[0].fill_between(bins[N], histograms[N], alpha=0.5, color='b')
        E, = ax[1].plot(iters, energies[:N], 'r-', marker='o', markersize=5)
        E_band = ax[1].fill_between(iters, energies[:N]-errors[:N], energies[:N]+errors[:N], alpha=0.5, color='r')
        x = jnp.linspace(xmin, xmax, 1000)
        V = potential(x)
        ax[0].plot(x, V - jnp.min(V), color='k', linestyle='dashed')

        if E_exact is not None:
            E_ref = ax[1].axhline(E_exact, color='k', linestyle='dashed', label='Exact')
            return (psi2, E, E_band, E_ref)

        else:
            return (psi2, E, E_band)

    anim = FuncAnimation(fig, drawframe, frames=int(nopt/skip), interval=100, blit=True)
    plt.close()

    return anim
