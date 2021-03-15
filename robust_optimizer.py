import cvxpy as cvx

class RobustOptimiser():
    def __init__(self, mu, sigma, kappa = 0.2, lambda_ = 4, omega):
        dim = mu.shape[0]
        w_rob = cvx.Variable(dim)
        erro_risk_term = cvx.quad_form(w_rob, omega)
        risk_aversion_term = cvx.quad_form(w_rob, sigma)
        obj = cvx.Maximize(
            mu @ w_rob + \
            kappa * cvx.sqrt(erro_risk_term) -\
            lambda_/2 * risk_aversion_term)
        
        return