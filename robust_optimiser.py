import cvxpy as cvx
import numpy as np 

class RobustOptimiser():
    def __init__(self, mu, sigma, omega, kappa, lambda_):
        self.w = self.calculate_weights(mu, sigma, omega, kappa, lambda_)

    def calculate_weights(self, mu, sigma, omega, kappa, lambda_):
        dim = mu.shape[0]
        w_rob = cvx.Variable(dim)
        Q = np.linalg.cholesky(omega)
        error_risk_term = cvx.norm(Q @ w_rob, 2)
        risk_aversion_term = cvx.quad_form(w_rob, sigma)
        obj = cvx.Maximize(w_rob @ mu - kappa * error_risk_term - lambda_/2 * risk_aversion_term)
        # obj = cvx.Maximize(w_rob @ mu)
        constr_1 = sum(w_rob)
        # problem = cvx.Problem(obj, [constr_1 == 0, risk_aversion_term <= lambda_, error_risk_term <= kappa])
        problem = cvx.Problem(obj, [constr_1 == 1])

        try:
            problem.solve(verbose = True)
        except:
            problem.solve(solver = 'SCS')
        if problem.value != np.inf:
            return w_rob.value
        else:
            print('The problem is unfeasible')
            return None

    def get_w_robust(self):
        return self.w