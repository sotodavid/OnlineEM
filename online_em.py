# 
import numpy as np
from scipy.stats import multivariate_normal

def initialize_parameters(num_clusters, x, y):
    n, T, d = x.shape
    theta = np.random.randn(num_clusters, d, d)
    gamma = np.random.randn(num_clusters, n, n, d, d)
    beta = np.random.randn(num_clusters, d, d)
    B = np.random.randn(num_clusters, d, d)
    Sigma = np.random.randn(num_clusters, d, d)
    # gamma matrix contains the interdependencies between xi and xj
    for c in range(num_clusters):
        for t in range(T):
            for i in range(n):
                for j in range(n):
                    gamma_ij = np.linalg.lstsq(np.array([x[i,t]]), np.array([x[j,t]]))[0]
                    gamma[c,i,j] = gamma_ij
    # We Make sure Sigma[i] are positive semi definite
    for i in range(len(Sigma)):
        Sigma[i] = np.dot(Sigma[i], Sigma[i].transpose())
    sigma_squared = np.random.randn(num_clusters, d, d)
    # We make sure sigma_squared[i] are positive semi definite
    for i in range(len(sigma_squared)):
        sigma_squared[i] = np.dot(sigma_squared[i], sigma_squared[i].transpose())
    u = np.random.randn(num_clusters, d)
    return theta, gamma, beta, B, Sigma, sigma_squared, u

def e_step(x, y, t, theta, gamma, beta, B, Sigma, sigma_squared, pi_c):
    n, T, d = x.shape
    p_c = np.zeros((n, len(pi_c)))
    mu_zxc = np.zeros((len(pi_c), n, T, d))
    Sigma_zxc = np.zeros((len(pi_c), d, d))
    for i in range(n):
        for c in range(len(pi_c)):
            # 1. We start by updating the cluster assignment probabilities
            mean_pred = theta[c] @ x[i, t] + beta[c] @ np.ones(d)
            likelihood = multivariate_normal.pdf(y[i, t], mean=mean_pred, cov=Sigma[c] + sigma_squared[c]) #np.diag(sigma_squared[c]))
            p_c[i, c] = pi_c[c] * likelihood

            # 2. We compute the posterior distribution's mean and covariance
            # N.B. : the posterior distribution is Gaussian.
            mu_zxc[c, i, t] = B[c].T @ np.linalg.pinv(B[c] @ B[c].T + Sigma[c]) @ x[i,t]
            Sigma_zxc[c] = np.eye(d) - B[c].T @ np.linalg.pinv(B[c] @ B[c].T + Sigma[c]) @ B[c]
        p_c[i, :] /= np.sum(p_c[i, :])
    return p_c, mu_zxc, Sigma_zxc

def m_step(x, y, t, mu_zxc, Sigma_zxc, theta, gamma, beta, B, Sigma, sigma_squared, p_c, u, num_clusters):
    n, T, d = x.shape
    pi_c_new = np.sum(p_c, axis=0) / (n)
    B_new = np.zeros_like(B)
    Sigma_new = np.zeros_like(Sigma)
    theta_new = np.zeros_like(theta)
    gamma_new = np.zeros_like(gamma)
    beta_new = np.zeros_like(beta)
    sigma_squared_new = np.zeros_like(sigma_squared)


    # Cluster assignment of x_i at time t
    cluster_xit = []
    for i in range(n):
        for c in range(num_clusters):
            if p_c[i,c] == np.max(p_c[i]) :
                cluster_xit.append(c)

    for c in range(num_clusters):
        # Computing B_new
        x_mu = np.array([p_c[i,c] * (x[i,t,:].reshape(d,1) @ mu_zxc[c,i,t].reshape(1,d)) for i in range(len(x))])
        B1 = np.sum(x_mu, axis = 0)
        weighted_Sigma_mu = np.array([p_c[i,c] * (Sigma_zxc[c] + mu_zxc[c,i,t].reshape(d,1) @ mu_zxc[c,i,t].reshape(1,d)) for i in range(len(x))])
        B2 = np.sum(weighted_Sigma_mu, axis = 0)
        B_new[c] = B1 @ np.linalg.pinv(B2)
        # Computing Sigma_new
        N_c = np.sum(p_c[:,c])
        Sigma_new[c] = np.cov(y.reshape(-1, d).T) * (1 / (N_c + 1e-6))
        # Computing theta_new
        theta1 = np.array([p_c[i,c] * (x[i,t,:].reshape(d,1) @ x[i,t,:].reshape(1,d)) for i in range(len(x))])
        theta1 = np.sum(theta1, axis = 0)
        theta2 = np.array([p_c[i,c] * (x[i,t,:].reshape(d,1) @ y[i,t,:].reshape(1,d)) for i in range(len(x))])
        theta2 = np.sum(theta2, axis = 0)
        theta_new[c] = np.linalg.pinv(theta1) @ theta2
        # Computing gamma_new
        # if cluster c has none or only one time series assigned to it, we don't update gamma
        if dict((a, cluster_xit.count(a)) for a in cluster_xit)[c] <= 1:
            for i in range(n):
                for j in range(n):
                    gamma_new[c,i,j] = gamma[c,i,j]
        else : # update gamma
            for i in range(n):
                for j in range(n):
                    g1 = np.array([p_c[i,c] * (x[j,t,:].reshape(d,1) @ x[j,t,:].reshape(1,d)) for j in range(len(x)) if j != i and cluster_xit[j] == c])
                    g1 = np.sum(g1, axis=0)
                    g2 = np.array([p_c[i,c] * (x[j,t,:].reshape(d,1) @ (y[i,t,:] - theta_new[c] @ x[i,t]).reshape(1,d)) for j in range(len(x)) if j != i and cluster_xit[j] == c])
                    g2 = np.sum(g2, axis=0)
                    gamma_new[c,i,j] = np.linalg.pinv(g1) @ g2
        # Computing beta_new
        beta1 = np.array([p_c[i,c] * (u[c].reshape(d,1) @ (y[i,t] - theta_new[c] @ x[i,t] - np.sum(np.array([gamma[c,i,j] @ x[j,t] for j in range(len(x)) if j != i and cluster_xit[j] == c]), axis=0)).reshape(1,d)) for i in range(len(x)) if cluster_xit[i] == c])
        beta1 = np.sum(beta1, axis=0)
        beta2 = np.sum(np.array([p_c[i,c] * u[c].reshape(d,1) @ u[c].reshape(1,d) for i in range(len(x)) if cluster_xit[i] == c]), axis=0)
        beta_new[c] = np.linalg.pinv(beta2) @ beta1
        # Computing sigma_squared_new
        theta_x = np.array([theta_new[c] @ x[i][t] for i in range(len(x))])
        sigma_squared_new[c] = np.cov(((y[:,t] - theta_x)**2)/n, rowvar=False)
        
    return pi_c_new, B_new, Sigma_new, theta_new, gamma_new, beta_new, sigma_squared_new

def optimal_control_learning(x, y, t, p_c, theta_new, gamma_new, beta, B_new, u, num_clusters):
    d = x.shape[2]
    u_c_new = np.zeros((num_clusters, d))
    for c in range(num_clusters):
        theta_x = np.array([theta_new[c] @ x[i][t] for i in range(len(x))])
        u1 = np.sum(p_c[:, c, None] * (y[:,t] - theta_x), axis=0)
        u2 = np.sum(p_c[:, c, None]) * (B_new[c].transpose() @ B_new[c])
        u_c_new[c] = np.linalg.pinv(u2) @ u1
    return u_c_new

def online_em_optimal_control(x, y, num_clusters):
    n, T, d = x.shape
    theta, gamma, beta, B, Sigma, sigma_squared, u = initialize_parameters(num_clusters, x.shape[2], x, y)
    pi_c = np.ones(num_clusters) / num_clusters
    for k in range(Sigma.shape[0]):
        Sigma[k] = np.dot(Sigma[k], Sigma[k].T)

    # Here starts the Online part !    
    for t in range(T):
        print("==========================================")
        print(f"Timestep {t+1}")
        p_c, mu_zxc, Sigma_zxc = e_step(x, y, t, theta, gamma, beta, B, Sigma, sigma_squared, pi_c)
        pi_c, B, Sigma, theta, gamma, beta, sigma_squared = m_step(x, y, t, mu_zxc, Sigma_zxc, theta, gamma, beta, B, Sigma, sigma_squared, p_c, u, num_clusters)
        u = optimal_control_learning(x, y, t, p_c, theta, gamma, beta, B, u, num_clusters)
        print("Optimal control for each cluster c:", u)
        print("Cluster priors pi_c =", pi_c)
        if t == (T - 1):
            final_p_c = p_c
    return theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u, final_p_c

def cluster_assign(x, p_c, num_clusters):
    cluster_assign = {} #dictionnary of final cluster assignments
    for c in range(num_clusters):
        for i in range(len(x)):
            if p_c[i][c] == np.max(p_c[i]):
                cluster_assign[f'x_{i}'] = c
    return cluster_assign

