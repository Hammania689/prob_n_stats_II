from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

class tablelookup_Error(Exception):
    pass

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

def t_score_lookup(samples, confidence, xBar, s, display=False):
    """
    Use to find confidence intervals for sample distributions n < 30.
    samples: values from random trails of a certain distribution
    confidence: interval of confidence (e.g .90, .95, etc)
    returns: confidence intervals for given sample distribution to represent a larger distribution
    """
    try:
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        n = len(samples)
        if n >= 30:
            raise tablelookup_Error("Use the Z score instead. Sample size is greater than 30!")
        # Lookup t value at given confidence interval
        t = scipy.stats.t.ppf((1 + confidence) / 2.0, n-1)

        # Calculate lower and upper boundaries 
        lowerBound = xBar - (t * (s / (n **.5)))
        upperBound = xBar + (t * ( s / (n **.5)))
        
        if display == True:
            print(f"T Score at {confidence} confidence: {t}")
            print(f"Mean is within range {lowerBound}, {upperBound} with {confidence} confidence!")
        return lowerBound, upperBound
    
    except ValueError:
        print("Confidence Interval must be a numeric value")
    except tablelookup_Error:
        print("Use the Z score instead. Sample size is greater than 30!")
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise
        
def z_score_lookup(samples, confidence, xBar, s, display=False):
    """
    Use to find confidence intervals for sample distributions n >= 30.
    samples: values from random trails of a certain distribution
    confidence: interval of confidence (e.g .90, .95, etc)
    returns: confidence intervals for given sample distribution to represent a larger distribution
    """
    try:
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        elif type(confidence) != int or type(confidence) != float:
            raise ValueError("Confidence Interval must be a numeric value")

        n = len(samples)
        if n < 30:
            raise tablelookup_Error()
        # Lookup t value at given confidence interval
        z = scipy.stats.norm.ppf((1 + confidence) / 2.0)

        # Calculate lower and upper boundaries 
        lowerBound = xBar - (z * (s / (n **.5)))
        upperBound = xBar + (z * ( s / (n **.5)))
        
        if display == True:
            print(f"Z Score at {confidence} confidence: {z}")
            print(f"Mean is within range {lowerBound}, {upperBound} with {confidence} confidence!")
        return lowerBound, upperBound
    
    except ValueError:
        print("Confidence Interval must be a numeric value")
    except tablelookup_Error:
        print("Use the T score instead. Sample size is less than 30!")
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise
        
def t_confidence_Interval_Difference_Of_Means(xSamples, ySamples, confidence):
    """
    xSamples = samples for r.v X
    ySamples = samples for r.v Y
    
    Assuming 
        1. m and n < 30
        2. Variance is unknown
    
    returns: Confidience Interval for difference between two means using T distribution
    """
    try:
        if len(xSamples) >= 30 or len(ySamples) >= 30:
            raise sampleSizeError("Should use normal distribution instead. m or n > 30.")
        
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        elif type(confidence) != int or type(confidence) != float:
            raise ValueError("Confidence Interval must be a numeric value")
    
        # Find mean and variance for both sample distributions
        n = len(xSamples) 
        xBar = sample_mean(xSamples)
        xSampStd = sample_variance(xSamples) ** .5
        
        m = len(ySamples)
        yBar = sample_mean(ySamples)
        ySampStd = sample_variance(ySamples) ** .5
        
        # Find t at alpha/2 and the new distribution's sample size - 2
        # Calculate the sample pooling standard deviation
        tAlpha = (1 + confidence) / 2.0
        t = scipy.stats.t.ppf(tAlpha, (m + n - 2)) 
        spsd = ((((n - 1)* (xSampStd**2)) + ((m - 1) * (ySampStd**2)))/(m + n - 2)) ** .5 
        
        # Find the lower and upper bound 
        # (X-Y) (+/-) t((spsd * (((1/m)+(1/n)) **.5))
        lowerBound = (xBar - yBar) - t * (spsd * (((1/m)+(1/n)) **.5))
        upperBound = (xBar - yBar) + t * (spsd * (((1/m)+(1/n)) **.5))
        
        return lowerBound, upperBound
    
    except sampleSizeError as inst:
        print(inst.args[0])
    
    except ValueError as inst:
        print(inst.args[0])

def prop_conf_interval(p, n, confidence):
    """
    p: point estimate of probability of success
    n: number of total trails
    confidence: degree of confidence interval contains mean of total distirbution  
    
    Return: Confidence Intervals for a binomail r.v with unkown p 
    """
    try:
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        elif type(confidence) != int and type(confidence) != float:
            raise ValueError("Confidence Interval must be a numeric value")
        
        z = scipy.stats.norm.ppf((1 + confidence) / 2.0)
        lowerBound = p - z * (((p * (1 - p)) / n) **.5)
        upperBound = p + z * (((p * (1 - p)) / n) **.5)

        return lowerBound, upperBound 
    except ValueError as inst:
        print (inst.args[0])
        print(type(confidence))
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise
        
def histogram_prop_conf_interval(p, n, confidence):
    """
    p: point estimate of probability of success
    n: number of total trails
    confidence: degree of confidence interval contains mean of total distirbution  
    
    Return: Confidence Intervals for a binomail r.v with unkown p 
    """
    try:
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        elif type(confidence) != int and type(confidence) != float:
            raise ValueError("Confidence Interval must be a numeric value")

        confidence = (1 + confidence) / 2.0
        z = scipy.stats.norm.ppf(confidence)

        lowerBound =  (p + (z **2)/(2 * n) - z * (((p * (1-p))/n + (z **2)/(4*n)**2) **.5)) / (1 + (z **2)/n)
        upperBound = (p + (z **2)/(2 * n) + z * (((p * (1-p))/n + (z **2)/(4*n)**2) **.5)) / (1 + (z **2)/n)
    
        return lowerBound, upperBound
    
    except ValueError as inst:
        print (inst.args[0])
        print(type(confidence))
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise
        
def biased_conf_interval(y, n, confidence):
    """
    y: frequency of event occuring
    n: number of total trails
    confidence: degree of confidence interval contains mean of total distirbution  
    
    Return: Confidence Intervals for a binomail r.v with unkown p 
    """
    try:
        if confidence > 1:
            confidence = confidence / 100.0
            print(f"Converting confidence interval to {confidence}")

        elif type(confidence) != int and type(confidence) != float:
            raise ValueError("Confidence Interval must be a numeric value")
        
        pBiased = (y + 2) / (n + 4)

        z = scipy.stats.norm.ppf((1 + confidence) / 2.0)

        lowerBound = pBiased - z * (((pBiased * (1 - pBiased) / (n + 4))) ** .5)
        upperBound = pBiased + z * (((pBiased * (1 - pBiased) / (n + 4))) ** .5)

        return lowerBound, upperBound
    
    except ValueError as inst:
        print (inst.args[0])
        print(type(confidence))
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        raise
    