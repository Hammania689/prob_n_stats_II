def sample_mean(samples):
  """
  samples: continous random variables over some distrubition 
  returns: Sample Mean
  """
  return (1/ len(samples)) * np.sum(samples)


def sample_variance(samples):
  """
  samples: continous random variables over some distrubition 
  returns: Sample Variance s^2 not v
  """
#   sample_mean_diff = samples - sample_mean(samples)
  return (1/(len(samples)-1)) * np.sum(((samples - sample_mean(samples)) **2))

def sample_histogram(samples, cls_bound):
    print(this,"still needs love")
    pass
# for i in range(len(cls_bound)):
#     interval = np.where((cls_bound[i - 1] < samples) & (samples < cls_bound[i]))
#     non_dup = np.unique(interval)
#     print(samples[interval])
#     print(samples[non_dup])