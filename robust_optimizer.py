import cvxpy as cvx
import numpy as np 

class RobustOptimiser():
    def __init__(self, mu, sigma, omega, kappa, lambda_):
        dim = mu.shape[0]
        w_rob = cvx.Variable(dim)
        Q = np.linalg.cholesky(omega)
        erro_risk_term = cvx.norm(Q*w_rob, 2)
        risk_aversion_term = cvx.quad_form(w_rob, sigma)
        obj = cvx.Maximize(mu * w_rob + kappa * erro_risk_term - lambda_/2 * risk_aversion_term)
        problem = cvx.Problem(obj, [])
        problem.solve()
        self.w = w_rob.value()
    def get_w_robust(self):
        return self.w

import pandas as pd
a = np.random.normal(0,1.2, 200)/100
b = np.random.normal(-1,1.9, 200)/100
c  = np.random.normal(3,0.05, 200)/100
d = np.random.normal(-1,1.9, 200)/100
e  = np.random.normal(2,0.05, 200)/100
df = pd.DataFrame({'A':a, 'B':b, 'C':c, 'D':d, 'E':e})
cov = df.cov().to_numpy()
returns = df.max(axis = 0).to_numpy()
m = RobustOptimiser(returns, cov, cov, 0.2, 4)
print(m.get_w_robust)