import random
from math import sqrt

def estimate_pi_monte_carlo(num_samples):
    count = 0
    for i in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            count += 1
    pi_estimate = 4 * count / num_samples
    return pi_estimate

def f(x):
    return sqrt(1 - x**2)

def estimate_pi_integration(num_steps):
    step_size = 1 / num_steps
    sum = 0
    for i in range(num_steps):
        x = (i + 0.5) * step_size
        sum += f(x)
    pi_estimate = 4 * step_size * sum
    return pi_estimate

def estimate_pi_series(num_terms):
    sum = 0
    for i in range(num_terms):
        term = ((-1) ** i) / (2 * i + 1)
        sum += term
    pi_estimate = 4 * sum
    return pi_estimate

num_samples = 1000000
pi_estimate_mc = estimate_pi_monte_carlo(num_samples)
print(f"Monte Carlo estimate of pi: {pi_estimate_mc:.10f}")
pi_estimate_int = estimate_pi_integration(num_samples)
print(f"Integration estimate of pi: {pi_estimate_int:.10f}")
pi_estimate_series = estimate_pi_series(num_samples)
print(f"Series estiamte of pi     : {pi_estimate_series:.10f}")
