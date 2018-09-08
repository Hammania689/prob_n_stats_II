
# coding: utf-8

# In[ ]:


import numpy as np
import math


# In[2]:


# Random Variables and all their possible values
y = np.arange(1, 5)
x = np.arange(1, 3)


# In[2]:


# Use to get the probability mass function at two specific random variables
def pmf(i, j):
    return (i + j) / 32

def marginal_prob(x, y):
    """
    x : is a single value
    y : is a array of all possible values other random function can take on
    
    returns: The Total probablity of X occuring (The total sum of X=x and Y= all possible values)
    """
    # Initial probability
    prob = 0
    
    # Find probabily of X=x and all of Y's possible discrete random values
    for i in np.nditer(y):
        prob += pmf(x, i)
    return prob

def mean(x, y, print_steps=False):
    """
    x : all possible discrete random variables of X. **Random Variable of Interest**
    y : all possible discrete random variables of Y.
    
    ** To get the Expected value of Y. Swap the placement of parameters when you call this function
    
    
    returns : Expected Value of random variable X
    """
    total_mean = 0
    
    for i in np.nditer(x):
        for j in np.nditer(y):
            # Print current index
            
            # Pmf for current x and y variables
            local_pmf = pmf(i , j)
            
            # Calculate expected value
            # Add to current index's expected value to total
            expected_val = i * local_pmf
            total_mean += expected_val
            
            if(print_steps):
                print(f"E[({i},{j})] : {expected_val}")
    if(print_steps):
        print (f"Expected value of {x} : {total_mean}")
    return total_mean

# Use to get the variance of a first array passed in
def variance(x, y, print_steps=False):
    """
    x : array of random variable values. Variance will be calculate for x
    y : array of accompanying random variable values in the bivariate distribution
    
    To get variance of Y simply swap the arguments when you call this function 
    returns : Variance of X
    """
    total_var = 0
    expected_val = mean(x, y)
        
    for i in np.nditer(x):
        # Find the total probability of event P(X=i) occurring
        
        for j in np.nditer(y):
            
            # Calculate the probability of P(X=i, Y=j)
            local_pmf = pmf(i , j)
            
            # Calculate variance 
            # var = ( x - mean)^2 * prob(X = i, Y = j) 
            # Add to total var
            var = (( i - expected_val) ** 2) * local_pmf
            total_var += var
            
            # Print current index and relevant output
            if(print_steps):
                print(f"\tP({i},{j}): {local_pmf}. Variance at ({i},{j}): {var}") 
    
    if(print_steps):
        print(f"Variance : {total_var}")
    return total_var

# Use to calculate the correlation of the two random variables
def covariance(x,y, print_steps=False):
    """
    x, y : an array of values that a discrete variable can take on
    return: covariance of x and y. 
    """
    # Store the mean of random variables X and Y 
    # Set the expected value to 0
    mean_x = mean(x,y)
    mean_y = mean(y,x)
    expected_val_xy = 0
    
    # Sum of E[(x - mean_x) * (y - mean_y)] for all possible combinations of x and y
    # E[(x - mean_x) * (y - mean_y)] = Summation of all x and y(x * y * f(x,y)) - (mean_x * mean_y)
    # Well the means of both X and Y are constant and can be pulled out in front of the Expected value
    
    # So first we focus on the summation of the joint pmf of X and Y 
    for i in np.nditer(x):
        for j in np.nditer(y):
            local_pmf = pmf(i, j)
            expected_val_xy += (i * j * local_pmf)
            if(print_steps):
                print(f"Pmf({i},{j}): {local_pmf} ")
                print(f"{expected_val_xy}")
        
    # Subtract the product of mean_x and mean_y here
    expected_val_xy -= (mean_x * mean_y)
    
    print(f"X Mean: {mean_x}, Y Mean: {mean_y}", f"\nCovariance: {expected_val_xy}")
    return expected_val_xy

def corr_coeffecient(x, y):
    """
    x,y : array of all values the discrete random variable can take on
    returns : the correlation coeffecient of X and Y
    """
    dev_x = math.sqrt(variance(x, y))
    dev_y = math.sqrt(variance(y, x))
    covar = covariance(x,y)
    
    p = covar / (dev_x * dev_y)
    print(f"X Dev:{dev_x}, Y Dev:{dev_y}\nP coefficient :{p}")
    return p

