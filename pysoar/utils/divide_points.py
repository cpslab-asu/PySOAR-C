import numpy as np

def pointsInTR(samples_in: np.array, samples_out:np.array, subregion: list) -> list:
    """

    Args:
        samples_in: Samples from Training set.
        samples_out: Evaluated values of samples from Training set.
        region_support: Min and Max of all dimensions

    Returns:
        list: Divided samples
    """    
    regionSamples = []
    corresponding_robustenss = []
    if samples_in.shape[0] == samples_out.shape[0] and  samples_out.shape[0] != 0:

        
        boolArray = []
        for dimension in range(len(subregion)):
            subArray = samples_in[:, dimension]
            logical_subArray = np.logical_and(subArray >= subregion[dimension, 0], subArray <= subregion[dimension, 1])
            boolArray.append(np.squeeze(logical_subArray))
        corresponding_robustenss = samples_out[(np.all(boolArray, axis = 0))]
        regionSamples = samples_in[(np.all(boolArray, axis = 0)),:]
    else:
        
        corresponding_robustenss = np.array([])
        regionSamples = np.array([[]])
            
    return regionSamples, corresponding_robustenss