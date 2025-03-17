#%%
import numpy as np

def initialize_parameters(n_x, n_z, n_u):
    theta = np.zeros_like(n_x)
    #beta = np.random.randn(num_clusters, num_features, num_features)
    A = np.random.randn(n_z, n_z)
    B = np.random.randn(n_x, n_z)
    C = np.random.randn(n_x, n_z)
    Q = np.random.randn(n_z, n_z)
    Sigma = np.random.randn(n_x, n_x)
    sigma2 = np.random.randn(1)
    beta = np.random.randn(n_u)
    theta = np.random.randn(n_x)
    bar_theta = np.random.randn(n_z)
    u = np.random.randn(n_u)
    return A, B, C, Q, Sigma, sigma2, theta, bar_theta, beta, u


def kalman_filter(Y, A, C, B, Q, Sigma, H, sigma2):
    """ Forward pass: Kalman Filtering"""
    T, _ = Y.shape
    n_x, n_z = C.shape[0], A.shape[0]
    
    # Initialize
    x_hat = np.zeros((T, n_x))
    z_hat = np.zeros((T, n_z))
    
    P_x = np.zeros((T, n_x, n_x))
    P_z = np.zeros((T, n_z, n_z))
    P_xz = np.zeros((T, n_x, n_z))  # Cross covariance

    # Initial estimates
    x_hat[0] = np.zeros(n_x)
    z_hat[0] = np.zeros(n_z)
    
    P_x[0] = np.eye(n_x)
    P_z[0] = np.eye(n_z)
    P_xz[0] = np.zeros((n_x, n_z))

    # Filtering loop
    for t in range(1, T):
        # Prediction step
        z_pred = A @ z_hat[t-1]
        x_pred = C @ x_hat[t-1] + B @ z_pred
        
        P_z_pred = A @ P_z[t-1] @ A.T + Q
        P_x_pred = C @ P_x[t-1] @ C.T + B @ P_z_pred @ B.T + Sigma
        P_xz_pred = C @ P_xz[t-1] @ A.T + B @ P_z_pred  # Cross covariance

        # Construct full covariance matrix
        P_joint = np.block([
            [P_x_pred, P_xz_pred],
            [P_xz_pred.T, P_z_pred]
        ])

        # Kalman Gain
        S = H @ P_joint @ H.T + sigma2  # Innovation covariance
        K = P_joint @ H.T @ np.linalg.inv(S)

        # Update step
        y_pred = H @ np.hstack([x_pred, z_pred])
        state_update = K @ (Y[t] - y_pred).flatten()

        x_hat[t] = x_pred + state_update[:n_x]
        z_hat[t] = z_pred + state_update[n_x:]

        # Update covariance
        P_update = (np.eye(n_x + n_z) - K @ H) @ P_joint
        P_x[t] = P_update[:n_x, :n_x]
        P_z[t] = P_update[n_x:, n_x:]
        P_xz[t] = P_update[:n_x, n_x:]

    return x_hat, z_hat, P_x, P_z, P_xz


def kalman_smoother(x_hat, z_hat, P_x, P_z, P_xz, A, C):
    """ Backward pass: Kalman Smoothing"""
    T = x_hat.shape[0]
    n_x, n_z = x_hat.shape[1], z_hat.shape[1]

    x_smooth = x_hat.copy()
    z_smooth = z_hat.copy()

    P_x_smooth = P_x.copy()
    P_z_smooth = P_z.copy()
    P_xz_smooth = P_xz.copy()

    for t in range(T-2, -1, -1):
        # Compute Smoothing Gain separately for x and z
        J_x = P_x[t] @ C.T @ np.linalg.inv(P_x[t+1])  # Smoothing gain for x
        J_z = P_z[t] @ A.T @ np.linalg.inv(P_z[t+1])  # Smoothing gain for z

        # State update using separate smoothing gains
        x_smooth[t] += J_x @ (x_smooth[t+1] - C @ x_hat[t])
        z_smooth[t] += J_z @ (z_smooth[t+1] - A @ z_hat[t])

        # Covariance update
        P_x_smooth[t] = P_x[t] + J_x @ (P_x_smooth[t+1] - P_x[t+1]) @ J_x.T
        P_z_smooth[t] = P_z[t] + J_z @ (P_z_smooth[t+1] - P_z[t+1]) @ J_z.T
        P_xz_smooth[t] = P_xz[t] + J_x @ (P_xz_smooth[t+1] - P_xz[t+1]) @ J_z.T

    return x_smooth, z_smooth, P_x_smooth, P_z_smooth, P_xz_smooth


def m_step(x_smooth, z_smooth, P_x_smooth, P_z_smooth, P_xz_smooth):
        # x and z dimensions
        dx1, dx2 = x_smooth.shape
        dz1, dz2 = z_smooth.shape
        # M-Step: Update parameters analytically
        A_new = (z_smooth[1:].reshape(dz2, dz1-1) @ z_smooth[:-1]) @ np.linalg.pinv(z_smooth[:-1].reshape(dz2, dz1-1) @ z_smooth[:-1])
        B_new = (x_smooth[1:].reshape(dx2, dx1-1) @ z_smooth[:-1]) @ np.linalg.pinv(z_smooth[:-1].reshape(dz2, dz1-1) @ z_smooth[:-1])
        C_new = (x_smooth[1:].reshape(dx2, dx1-1) @ x_smooth[:-1]) @ np.linalg.pinv(x_smooth[:-1].reshape(dx2, dx1-1) @ x_smooth[:-1])

        # Q_new
        z_Az = np.array([z_smooth[1:][i] - A_new @ z_smooth[:-1][i] for i in range(len(z_smooth[1:]))])
        Q_new = ((z_Az.reshape(dz2, dz1-1)) @ z_Az) / (T-1)
        # Sigma_new
        x_Cx_Bz = np.array([x_smooth[1:][i] - C_new @ x_smooth[:-1][i] - B_new @ z_smooth[:-1][i] for i in range(len(x_smooth[1:]))])
        Sigma_new = (x_Cx_Bz.T @ x_Cx_Bz) / (T-1)
        # theta_new
        xy = np.array([x_smooth[i] * Y[i] for i in range(len(Y))])
        xx = np.array([x_smooth[i] @ x_smooth[i].T for i in range(len(x_smooth))])
        theta_new = np.sum(xy, axis=0) * (1/np.sum(xx, axis=0))
        # bar_theta_new
        zy = np.array([z_smooth[i] * Y[i] for i in range(len(Y))])
        zz = np.array([z_smooth[i] @ z_smooth[i].T for i in range(len(z_smooth))])
        bar_theta_new = np.sum(zy, axis=0) * (1/np.sum(zz, axis=0))
        # beta_new
        uy = np.array([z_smooth[i] * Y[i] for i in range(len(Y))])
        uu = np.array([z_smooth[i] @ z_smooth[i].T for i in range(len(z_smooth))])
        beta = np.sum(uy, axis=0) * (1/np.sum(uu, axis=0))
        # sigma2_new
        e = np.array([Y[i] - theta_new @ x_smooth[i] - bar_theta_new @ z_smooth[i] for i in range(len(x_smooth))])
        sigma2_new = np.mean(e, axis=0)

        return A_new, B_new, C_new, Q_new, Sigma_new, sigma2_new, beta, theta, bar_theta, gamma


def optimal_control_learning(x, y, p_c, t, theta_new, gamma_new, beta, B_new, u, n_clusters):
    d = x.shape[2]
    u_c_new = np.zeros((n_clusters, d))
    for c in range(n_clusters):
        theta_x = np.array([theta_new[c] @ x[i][t] for i in range(len(x))])
        u1 = np.sum(p_c[:, c, None] * (y[:,t] - theta_x), axis=0)
        u2 = np.sum(p_c[:, c, None]) * (B_new[c].transpose() @ B_new[c])
        u_c_new[c] = np.linalg.pinv(u2) @ u1
    return u_c_new

#%%
#Em algorithm

def em_algorithm(Y, X, A, C, B, Q, Sigma, H, sigma2):
    """ Expectation-Maximization (EM) Algorithm for Gaussian State-Space Model """
    n, T, d = X.shape
    # Initialize the parameters
    A, B, C, Q, Sigma, sigma2, theta, bar_theta, beta, u = initialize_parameters(n_x, n_z, n_u)


    for t in range(T):
        print("==========================================")
        print(f"Timestep {t+1}")
        # E-Step: Kalman Filtering & Smoothing
        x_hat, z_hat, P_x, P_z, P_xz = kalman_filter(Y, A, C, B, Q, Sigma, H, sigma2)
        x_smooth, z_smooth, P_x_smooth, P_z_smooth, P_xz_smooth = kalman_smoother(x_hat, z_hat, P_x, P_z, P_xz, A, C)
        # M-Step: Update parameters analytically
        A_new, B_new, C_new, Q_new, Sigma_new, sigma2_new, beta_new, theta_new, bar_theta_new, gamma_new = m_step(x_smooth, z_smooth, P_x_smooth, P_z_smooth, P_xz_smooth)
        # Control Step
        u = optimal_control_learning(X, Y, t, theta, gamma, beta, B, u)
        print(f"Timestep {t+1}: Log-likelihood {np.linalg.norm(Y - theta_new @ x_smooth)}")
        A, B, C = A_new, B_new, C_new
        Q, Sigma, sigma2 = Q_new, Sigma_new, sigma2_new
        theta, bar_theta, beta, gamma = theta_new, bar_theta_new, beta_new, gamma

    return A, B, C, Q, Sigma, sigma2, theta, bar_theta, beta, gamma
