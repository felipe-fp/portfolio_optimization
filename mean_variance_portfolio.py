import cvxpy as cvx

class MVO():
    def __init__(self, mu, sigma, min_returns):
        self.w = self.calculate_weights(mu, sigma, min_returns)

    def calculate_weights(self, mu, sigma, min_returns):
        dim = mu.shape[0]
        w_opt = cvx.Variable(dim)
        obj = cvx.Minimize(cvx.quad_form(w_opt, sigma))
        constr_1 = w_opt @ mu
        constr_2 = sum(w_opt)
        problem = cvx.Problem(obj, [constr_1 >= min_returns, constr_2 == 1])
        try:
            problem.solve(verbose = False)
        except:
            problem.solve(solver = 'SCS')
        w = w_opt.value
        return w
    def get_weights(self):
        return self.w
