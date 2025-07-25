import math, numpy as np
# implemented for practice
def mean(nums):
    return sum(nums) / len(nums)

def std(nums):
    avg = mean(nums)
    variance = sum([(num - avg)**2 for num in nums]) / (len(nums) - 1)
    std = math.sqrt(variance)
    assert(abs(std - np.std(nums, ddof=1)) < .01)
    return std