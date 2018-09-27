import numpy as np
import gym
import gym.spaces.box as gym_box
from gym.utils import seeding
import scipy.linalg as LA
import autograd.numpy as anp
from autograd import grad


class LQREnv(gym.Env):

    def __init__(self, x_dim=100, u_dim=10, seed=100, T=10, noise_cov=0.01):
      super(LQREnv, self).__init__()
      self.np_random, seed = seeding.np_random(seed)
      
      self.A = np.eye(x_dim) * 0.9
      for i in range(x_dim):
        for j in range(x_dim):
          if i == j:
            if i > 0 and i < x_dim-1:
              self.A[i-1, j] = self.A[i+1, j] = 0.01
            elif i == 0:
              self.A[i+1, j] = 0.01
            else:
              self.A[i-1, j] = 0.01
          else:
            continue
      
      self.B = np.ones((x_dim, u_dim))
      self.Q = np.eye(x_dim) / 1000
      self.R = np.eye(u_dim)
      
      self.x_dim = x_dim
      self.a_dim = u_dim
      
      self.observation_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (x_dim, ), dtype=np.float32)
      self.action_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (u_dim, ), dtype=np.float32)
      
      self.init_state = self.np_random.normal(0, 1, size=x_dim)
      
      self.state = self.init_state.copy()
      self.noise_cov = noise_cov
      
      self.T = T
      
      def cost(w):
        total_c = 0
        x = anp.copy(self.init_state)
        for i in range(self.T):
          u = anp.dot(w, x)  # w.dot(x)
          total_c += anp.dot(x, anp.dot(self.Q, x)) + anp.dot(u, anp.dot(self.R, u)) 
          x = anp.dot(self.A, x) + anp.dot(self.B, u) + self.np_random.normal(0, self.noise_cov, size=x_dim)
        return total_c

      self.grad_func = grad(cost)
      
      self.reset()
      
    def reset(self):
      self.state = self.np_random.normal(0, 1, size=self.x_dim)  # self.init_state.copy()
      self.t = 0
      return self.state

    def step(self, a): 
      
      cost = self.state.dot(self.Q).dot(self.state) + a.dot(self.R).dot(a)
      next_state = np.dot(self.A, self.state) + np.dot(self.B, a) + self.np_random.normal(0, self.noise_cov, size=self.x_dim)
      self.state = next_state.copy()

      done = False
      self.t += 1
      if self.t >= self.T:
        done = True
      return self.state, -cost, done, None

    def render(self, mode='human', close=False):
      pass 

    def evaluate_policy(self, K, non_stationary=False):
      gradient_of_K = self.grad_func(K)
      return gradient_of_K
