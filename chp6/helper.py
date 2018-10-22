from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_sample_data(chp,sect,prob, display=False):
    """
    chp,sect,prob: Numbers for chapter, section and problem
    display: Print content once loaded
    returns: Loaded problem sample data, if present
    """
    
    file_name = 'E'+ str(chp)+ '_'+ str(sect)+ '-'+ "{0:0=2d}".format(prob)+ '.txt'
    file_path =Path(Path.cwd().parents[0],'prob_data',file_name)
    if Path.exists(file_path):
        print(f"Loaded {file_name} sucessfully.")
        content = pd.read_csv(file_path,delimiter=' ',header=None).values.flatten()
        if(display):
            print(content)
        return content
    print(f"Could not find {file_name}...")
    


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


def mle(samples):
    """
    Same as sample_mean
    samples: values from random trails of a certain distribution
    returns: theta(=X bar), expected value for the sample size
    """
    n = len(samples)
    return (1/n) * np.sum(samples)

def mme_geo(samples, moment=1):
    """
    Use for geometric point estimation
    samples: values from random trails of a certain distribution
    returns: p~(=1/(X bar)), expected value for the method of moments 
    """
    samples = samples ** moment
    k = len(samples)
    return ( k / np.sum(samples))