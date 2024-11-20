from measurement2 import quantum_measurement

beta2 = 1.5707963         # 输入 beta2
cos_mu1 = 0.9021       # 输入 cos(mu1)
cos_mu2 = 0.9021       # 输入 cos(mu2)
theta = 0.4        # 输入 theta
alpha = 1.28043         # 输入 alpha

results = quantum_measurement(beta2, cos_mu1, cos_mu2, theta, alpha)
print(results)
