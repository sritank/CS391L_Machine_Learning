import numpy as np
import scipy.io
import ica_functions
import matplotlib.pyplot as plt

# Load the raw data
file_in = scipy.io.loadmat('sounds.mat')
raw_sounds = np.float32(file_in['sounds'])

# Set the frequency (from the assignment, depends on input data)
FREQUENCY = 11025

# Set which of the sounds we want and get the matrix for that
sounds_to_use = [0, 1, 2]
mat_U = raw_sounds[sounds_to_use,:]

# Normalize each row of mat_U between -1 and 1
mat_U = ica_functions.normalizeMatrix(mat_U)

# Set the number of mixed signals
m = 3

# Listen to the input sounds
ica_functions.playSound(raw_sounds[0,:], FREQUENCY) # homer thermodynamics
ica_functions.playSound(raw_sounds[1,:], FREQUENCY) # machinery
ica_functions.playSound(raw_sounds[2,:], FREQUENCY) # applause
ica_functions.playSound(raw_sounds[3,:], FREQUENCY) # laughter
ica_functions.playSound(raw_sounds[4,:], FREQUENCY) # white noise/fireworks

# Make sure m >= n
m = max([m, len(sounds_to_use)])

# Generate the mixer matrix A (by loading)
mat_A = np.load('ica_once_A.npy')

# Get the mixed signal matrix X
mat_X = mat_A.dot(mat_U)

# Listen to the mixed signals
for i in range(m):
	ica_functions.playSound(mat_X[i,:], FREQUENCY)

# Initialize W (by loading)
W_init = np.load('ica_once_W0.npy')

# Do the ICA 
[W, Y, iterations] = ica_functions.performICA(W_init, mat_X, 0.01, 10000, 0.0000001)
print('Number of iterations was', iterations)

# Normalize the output
Y = ica_functions.normalizeMatrix(Y)

# Listen to the unmixed signals
for i in range(len(sounds_to_use)):
	ica_functions.playSound(Y[i,:], FREQUENCY)

# Pick a range for plotting. Showing 40000 data points is too much
t0 = 10000
t1 = 10300

# Show how close the unmixed sounds were
[norms, locs] = ica_functions.findNearestSignal(Y, mat_U)
print(norms)

# Plot the sounds
ica_functions.plotCoorespondingSignals(Y, 'ICA', mat_U, 'Original', t0, t1)
plt.show()