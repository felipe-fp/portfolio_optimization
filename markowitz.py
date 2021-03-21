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
        constr_3 = sum(cvx.abs(w_opt))
        problem = cvx.Problem(obj, [contr_1 <= max_var, constr_2 == 0])
        try:
            problem.solve(verbose = True)
        except:
            problem.solve(solver = 'SCS')
        w = w_opt.value

        return w
    def get_weights(self):
        return self.w

# a = np.random.normal(0,1.2, 200)/100
# b = np.random.normal(-1,1.9, 200)/100
# c  = np.random.normal(3,0.05, 200)/100
# d = np.random.normal(-1,1.9, 200)/100
# e  = np.random.normal(2,0.05, 200)/100
# df = pd.DataFrame({'A':a, 'B':b, 'C':c, 'D':d, 'E':e})
# from sigma import *
# from mu import *
# obj1 = Sigma(df, 1)
# s = obj1.get_sigma()
# obj2 = mu(df)
# r = obj2.get_expected_returns()
# # x = np.random.normal(0,1.2, 5)
# m = Markowitz(r, s, 20)


# # cov = df.cov().to_numpy()
# # returns = df.max(axis = 0).to_numpy()
# # print(cov)
# # print(returns)
# # m = Markowitz(returns, cov, 20)
# print(m.get_weights())


# TODO:
# dividir dataframe of mes pra rebalancear
# corrigir markowitz e robust