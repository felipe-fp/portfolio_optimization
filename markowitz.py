import numpy as np
import pandas as pd
import cvxpy as cvx

class Markowitz():
    def __init__(self, mu, sigma, max_var):
        self.w = self.calculate_weights(mu, sigma, max_var)

    def calculate_weights(self, mu, sigma, max_var):
        dim = mu.shape[0]
        w_opt = cvx.Variable(dim)
        obj = cvx.Maximize(w_opt @ mu)
        contr_1 = cvx.quad_form(w_opt, sigma)
        constr_2 = sum(w_opt)
        # problem = cvx.Problem(obj, [contr_1 <= max_var, constr_2 == 1])
        problem = cvx.Problem(obj, [contr_1 <= max_var])
        try:
            problem.solve(verbose = True)
        except:
            problem.solve(solver = 'SCS')
        w = w_opt.value

        return w
    def get_weights(self):
        return self.w