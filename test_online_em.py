import numpy as np
from online_em import online_em_optimal_control

def test_without_control(x,y,num_clusters):
    """Run the online EM algorithm without control."""
    theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = online_em_optimal_control(x, y, num_clusters)
    print("Test without control completed.")
    return theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c

def test_with_evolving_control(x,y,num_clusters):
    """Run the online EM algorithm with an evolving control."""
    theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = online_em_optimal_control(x, y, num_clusters)
    print("Test with evolving control completed.")
    return theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c

def test_with_autoregressive_control(x,y,num_clusters):
    """Run the online EM algorithm with autoregressive control."""
    theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = online_em_optimal_control(x, y, num_clusters)
    print("Test with autoregressive control completed.")
    return theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c

def cluster_assignment(x, p_c, gt_clusters, num_clusters):
    cluster_assign = {} #dictionnary of final cluster assignments
    for c in range(num_clusters):
        for i in range(len(x)):
            if p_c[i][c] == np.max(p_c[i]):
                cluster_assign[f'x_{i}'] = c

    count = 0
    for i in range(len(x)):
        if cluster_assignment[f'x_{i}'] == gt_clusters[i]:
            count += 1
    print("Rate of correct cluster assignments :", count/len(x))
    return cluster_assign

# Defining the data 
def generate_data(n, T, d, n_clusters, pi_c, control_type):
    '''
    (n,T,d) : dimensions of the data x
    n_clusters : number of clusters
    pi_c : cluster assignment probabilities
    control_type : Refers to the type of the control u we want to use for generating the data
                   There are three type :
                        - "No control"
                        - "Evolving control"
                        - "autoregressive control"
    
    The function returns the data x_it and y_it of shape (n,T,d) and the ground-truth cluster assignments c_i
    '''


    # Model parameters
    B_c = [np.random.randn(d, d) for _ in range(n_clusters)]  # Latent-to-explanatory mapping
    Sigma_c = [np.eye(d) * 0.1 for _ in range(n_clusters)]  # Covariance of noise eta_i
    theta_c = [np.random.randn(d,d) for _ in range(n_clusters)]  # Coefficients for x_i
    beta_c = [np.random.randn(d,d) for _ in range(n_clusters)]  # Control coefficients
    sigma_eps = 0.1  # Noise variance in y
    # Cluster-level control
    u_c = np.random.randn(n_clusters, T, d)

    if control_type == "Evolving control" :
        # Evolving Cluster-Level Control
        # We use a simple u_t = u_o + alpha * t evolving control
        u_c = np.random.randn(n_clusters, T, d)
        for c in range(n_clusters):
            u_0 = np.random.randn(1, d)
            alpha = np.array([0.05, -0.03])
            for t in range(T):
                u_c[c,t] = u_0 + alpha * t
    
    elif control_type == "autoregressive control" :
            u_c = np.zeros((n_clusters, T, d))
            rho = 0.9  # Persistence factor
    
            for t in range(T):
                epsilon = np.random.randn(n_clusters, d) * 0.1  # Small noise
                u = rho * u + epsilon  # AR(1) update

    elif control_type == "No control" :
        u_c = np.zeros((n_clusters, T, d))


    # Generate data
    c_i = np.random.choice(n_clusters, size=n, p=pi_c)  # Assigning clusters to time series x_i
    z_it = np.random.randn(n, T, d)  # Sample latent variables
    x_it = np.zeros((n, T, d))
    y_it = np.zeros((n, T, d))

    for i in range(n):
        c = c_i[i]  # Cluster of time series i
        for t in range(T):
            # Generate variables x_it
            x_it[i, t] = B_c[c] @ z_it[i, t] + np.random.multivariate_normal(np.zeros(d), Sigma_c[c])
            # Generate variables y_it
            interaction_term = sum(np.random.randn() * x_it[j, t] for j in range(n) if j != i and c_i[j] == c)
            y_it[i, t] = theta_c[c] @ x_it[i, t] + interaction_term + beta_c[c] @ u_c[c, t] + np.random.normal(0, sigma_eps)

    return x_it, y_it, c_i

# Parameters
n = 100  # Number of time series
T = 48  # Number of time steps
n_clusters = 3 
d = 2  # Dimension of variables
# Cluster assignment probabilities
pi_c = np.array([0.5, 0.3, 0.2])
control_type = "autoregressive control"

x, y, c_i = generate_data(n, T, d, n_clusters, pi_c, control_type)

# Run tests
if __name__ == "__main__": 
    if control_type == "No control":
        theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = test_without_control(x,y,n_clusters)
        cluster_assign = cluster_assignment(x, p_c, c_i, n_clusters)
    elif control_type == "Evolving control":
        theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = test_with_evolving_control(x,y,n_clusters)
        cluster_assign = cluster_assignment(x, p_c, c_i, n_clusters)
    elif control_type == "autoregressive control":
        theta, gamma, beta, B, Sigma, sigma_squared, pi_c, u_c_new, p_c = test_with_autoregressive_control(x,y,n_clusters)
        cluster_assign = cluster_assignment(x, p_c, c_i, n_clusters)