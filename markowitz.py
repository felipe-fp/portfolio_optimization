import numpy as np
import pandas as pd
class Markowitz():
    def __init__(self, mu, sigma, max_var):
        self.w = self.calculate_weights(mu, sigma, max_var)
    def calculate_weights(self, mu, sigma, max_var):
        mu_t = mu.reshape(mu.shape[0],1)
        cov_inv = np.linalg.inv(sigma)
        lambda_2 = (mu.dot(cov_inv.dot(mu_t)))/(4*max_var)
        lambda_ = lambda_2[0] ** 0.5
        w_opt = cov_inv.dot(mu_t)/(2*lambda_)
        return w_opt
    def get_weights(self):
        return self.w

# a = np.random.normal(0,1.2, 1000)
# b = np.random.normal(-1,1.9, 1000)
# c  = np.random.normal(2,0.5, 1000)
# df = pd.DataFrame({'A':a, 'B':b, 'C':c})
# from sigma import *
# from mu import *
# obj1 = Sigma(df, 1)
# s = obj1.get_sigma()
# print(s)
# obj2 = mu(df)
# x = obj2.get_expected_returns()
# m = Markowitz(x, s, 1)
# print(m.get_weights())