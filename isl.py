import midi
import numpy as np
from data import *
from midi_to_statematrix import *
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import sys, os


# Function takes the full state matrix and returns
# a list of individual measures
def get_measure_groups(statematrix):
    batch_size = 16*1
    measures = []

    n_ticks = len(statematrix)
    
    for i in range(int(np.floor(n_ticks / float(batch_size)))):
        measures.append(statematrix[i*batch_size:(i+1)*batch_size])
        
    return measures

# Takes a list of generated midi filenames as input, converts the
# files to state matricies, converts the state matricies to lists
# of measures and returns the density estimate
def get_KDE(generated_list):
    sm_tot = midiToNoteStateMatrix(generated_list[0])

    for f in generated_list[1:]:
        sm_tot += midiToNoteStateMatrix(f)
        
    mgs = get_measure_groups(sm_tot)

    print np.shape(mgs)
    
    # Double the size of mgs so that we have more measures
    # than dimensions. You may need to increase the number 
    # if you don't have enough generated samples. (kinda-hacky but works)
    mgs = mgs*2

    KDE = KDEMultivariate(data=mgs, var_type='u'*78*16*2)
    
    return KDE

# Takes the density estimate (from get_KDE), and a list of midi filenames
# for the test set, and returns the a list containing the ISL
# for each measure of each file of the test set. 
def get_ISL(dens, test_list):
    sm_test = midiToNoteStateMatrix(test_list[0])
    
    for f in test_list:
        sm_test += midiToNoteStateMatrix(f)
        
    test_mgs = get_measure_groups(sm_test)

    ISLs = []
    
    for measure in test_mgs:
        test_measure_flat = np.reshape(measure, 78*16*2)
        measure_ISL = np.log(dens.pdf(test_measure_flat))
        ISLs.append(measure_ISL)
        
    return ISLs

if __name__ == "__main__":
    # Get the list of generated files (all start with 'composition')
    generated_list = ['output/' + f for f in os.listdir('output/') if 'composition' in f]

    # Compute density estimate
    dens = get_KDE(generated_list)

    # Get list of bach files to use as test set
    test_list = ['music/' + f for f in os.listdir('music/') if 'bach' in f and 'zip' not in f]

    # Compute ISL of test set
    bach_ISL = get_ISL(dens, test_list)

    # get list of ISL which arent NAN... 
    non_nan_ISLs = [f for f in bach_ISL if not np.isnan(f)]
    
    # Print average of the non-NAN ISL's as the overall ISL
    print "Mean non-nan ISL: {}".format(np.mean(non_nan_ISLs))
