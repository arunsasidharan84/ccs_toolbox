def generatepinknoise(nsamples, nlevels=16):
    """
    Generating pink noise 
    using stochastic Voss-McCartney algorithm
    
    The fundamental idea of this algorithm is to add up several 
    sequences of random numbers that get updated at different rates. 
    The first source should get updated at every time step; 
    the second source every other time step, 
    the third source ever fourth step, and so on.
    
    Based on the code by Allen Downey at:
        https://www.dsprelated.com/showarticle/908.php
    
    Created on Thu Apr  2 17:24:51 2020
    
    @author: Arun Sasidharan
    """
    
    import numpy as np
    import pandas as pd
    
    # Start with an array with one row per timestep and 
    # one column for each of the white noise sources.
    array = np.empty((nsamples, nlevels))
    array.fill(np.nan)
    array[0, :] = np.random.random(nlevels)
    array[:, 0] = np.random.random(nsamples)
    
    # Choose the locations where the random sources change
    # If the number of samples is n, the number of changes in 
    # the first column is n, the number in the second column is n/2 
    # on average, the number in the third column is n/4 on average, etc.
    cols = np.random.geometric(0.5, nsamples)
    
    # If we generate a value out of bounds, we set it to 0 
    # (so the first column gets the extras)
    cols[cols >= nlevels] = 0
    
    # Within each column, we choose a random row from a 
    # uniform distribution
    rows = np.random.randint(nsamples, size=nsamples)
    
    # Put random values at rach of the change points
    array[rows, cols] = np.random.random(nsamples)
    
    # Do zero-order hold to fill in the NaNs using pandas dataframe
    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)
    
    pinknoise = total.values-(nlevels/2)
    return pinknoise
    