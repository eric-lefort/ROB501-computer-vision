import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C, WhiteKernel


# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T


DO_DEPTH = True
KNOWN_DEPTH = not DO_DEPTH

for i in range(3,4):
    INIT = int(i)
    if INIT == 0:
        TITLE = "Base init, "
        C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
        t_init = np.array([[-0.2, 0.3, -5.0]]).T
        
        gain_lower = 0.05
        gain_upper = 2.0
        N = 20  # Number of iterations

    elif INIT == 1:
        TITLE = "Hard init 1, "
        C_init = dcm_from_rpy([np.pi/10, -np.pi, -np.pi/12])
        t_init = np.array([[-0.1, 0.3, 5.0]]).T

        gain_lower = 0.05
        gain_upper = 0.8
        N = 30  # Number of iterations

    elif INIT == 2:
        TITLE = "Hard init 2, "
        C_init = dcm_from_rpy([np.pi/10, -np.pi/9, np.pi])
        t_init = np.array([[-0.1, 0.3, -15.0]]).T
        
        gain_lower = 0.05
        gain_upper = 0.2
        N = 50  # Number of iterations

    elif INIT == 3:
        TITLE = "Hard init 3, "
        C_init = dcm_from_rpy([np.pi/10, -np.pi/9, np.pi*11/12])
        t_init = np.array([[-0.1, 0.3, -15.0]]).T
        
        gain_lower = 0.05
        gain_upper = 0.2
        N = 100  # Number of iterations
    
    if KNOWN_DEPTH:
        TITLE += "known depth"
    else:
        TITLE += "approximate depth"
    Twc_last = np.eye(4)
    Twc_last[0:3, :] = np.hstack((C_last, t_last))
    Twc_init = np.eye(4)
    Twc_init[0:3, :] = np.hstack((C_init, t_init))




    def evaluate_gain(gain):
        try:
            return ibvs_simulation(Twc_init, Twc_last, pts, K, gain, do_depth=DO_DEPTH, do_plot=False)
        except Exception as e:
            print(e)
            return 150  # A large penalty for invalid evaluations
        

    kernel = 1 * RBF(length_scale=1, length_scale_bounds=(1e-2, 5)) \
         + 0.02 * RationalQuadratic(length_scale=0.01, alpha=0.1, length_scale_bounds=(1e-3, 5)) \
         + 1 * WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1)

    # Initial points (lower and upper bounds for gains)
    X = np.array([gain_lower, gain_upper]).reshape(-1, 1)
    y = np.array([evaluate_gain(x[0]) for x in X]).reshape(-1, 1)

    gp.fit(X, y)

    # EI
    def expected_improvement(X_sample, y_sample, model, xi=0.1):
        X_sample = np.atleast_2d(X_sample)
        mu, sigma = model.predict(X_sample, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.min(y_sample)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    # get next sampling point
    def propose_location(acquisition, X_sample, y_sample, model, bounds):
        def min_obj(X):
            return -acquisition(X.reshape(-1, 1), y_sample, model)

        # Propose location using optimization
        res = minimize(
            min_obj,
            x0=np.random.uniform(*bounds, size=(1,)),
            bounds=[bounds],
            method="L-BFGS-B"
        )
        return res.x

    bounds = (gain_lower, gain_upper)

    for _ in range(N):
        print(bounds)
        x_next = propose_location(expected_improvement, X, y, gp, bounds)
        y_next = evaluate_gain(x_next)
        
        print(x_next, y_next)

        X = np.vstack((X, x_next.reshape(-1, 1)))
        y = np.vstack((y, [[y_next]]))
        
        gp.fit(X, y)

    X_eval = np.linspace(gain_lower, gain_upper, 400).reshape((-1, 1))
    mean_prediction, std_prediction = gp.predict(X_eval, return_std=True)

    plt.scatter(X, y, label="Observations")
    plt.plot(X_eval, mean_prediction, label="Gaussian Regression Predictor")
    plt.fill_between(
        np.array(X_eval).ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("gain")
    plt.ylabel("Number of iterations")
    # plt.ylim(top=200)
    plt.ylim([-10, 200])
    _ = plt.title(TITLE + f", {N} samples")
    plt.savefig("plots/plot_" + "_".join(TITLE.lower().replace(",", "").split(" ")) + ".png", dpi=300)
    plt.show()
    with open("output.log", "a") as f:
        print(f"{TITLE}", file=f)
        idx = np.argmin(y)
        print(f"Best sampled gain: {X[idx, 0]}, \tn interations: {y[idx, 0]}", file=f)