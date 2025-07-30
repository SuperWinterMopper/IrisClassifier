import math, numpy as np
from scipy.stats import norm

# implemented for practice
def mean(nums):
    return sum(nums) / len(nums)

def std(nums):
    avg = mean(nums)
    variance = sum([(num - avg)**2 for num in nums]) / (len(nums) - 1)
    std = math.sqrt(variance)
    assert(abs(std - np.std(nums, ddof=1)) < .01)
    return std

def gaussianPDF(x, u, std):
    return 1 / (std * math.sqrt(2 * np.pi)) * np.exp(-.5 * ((x - u) / std)**2)