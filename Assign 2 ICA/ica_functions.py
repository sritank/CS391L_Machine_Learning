import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
# import ipdb;

# Inputs: Sound data (1xn), sampling frequency (scalar)
def playSound(sound, freq):
	# Normalize the sound data and play it
	sd.play(normalizeData(sound), freq, blocking=True)
	return True

# Normalize the sound data [-1, 1] for playing or graphing
def normalizeData(sound):
	return 2.*(sound - np.min(sound))/np.ptp(sound)-1

# Performs normalizeData on each row and returns it
def normalizeMatrix(data):
	output = data
	for i in range(data.shape[0]):
		output[i,:] = normalizeData(output[i,:])
	return output

# Inputs: the initial W matrix (n x m)
# X is the noisy data (m x t), alpha the descent contant
# max_n the maximum iterations, epsilon the convergence limit
# Returns W, Y, and the number of iterations run
def performICA(W0, X, alpha, max_n, epsilon):
	# Initialize Y and Z to size (n x t)
	num_signals = W0.shape[0]
	num_t = X.shape[1]
	mat_Y = np.zeros((num_signals, num_t))
	mat_Z = np.zeros((num_signals, num_t))
	# ipdb.set_trace()
	# Initialize dW to size (m x t)
	dW = np.zeros(W0.shape)

	# Set up the loop variables
	couter = 0
	mat_W = W0
	convergence = 2*epsilon

	while(couter < max_n and convergence > epsilon):
		# Find our estimation of the unmixed sounds
		mat_Y = mat_W.dot(X)

		# Construct Z
		mat_Z = 1/(1+(np.exp(-1*mat_Y)))

		# Find dW, amount we want to change by
		dW = alpha * ((np.eye(num_signals) + (1-2*mat_Z).dot(np.transpose(mat_Y))/X.shape[1]).dot(mat_W))

		# Increment W
		mat_W += dW

		# Count this iteration
		couter += 1

		# Get how close we are to convergence
		convergence = np.linalg.norm(dW)/np.linalg.norm(mat_W)

	return [mat_W, mat_W.dot(X), couter]

# Same as performICA but does N iterations, no epsilon check
def performICAforN(W0, X, alpha, max_n):
	# Initialize Y and Z to size (n x t)
	num_signals = W0.shape[0]
	num_t = X.shape[1]
	mat_Y = np.zeros((num_signals, num_t))
	mat_Z = np.zeros((num_signals, num_t))

	# Initialize dW to size (m x t). Ensure the norm > epsilon
	dW = np.ones(W0.shape)

	# Set up the loop variables
	couter = 0
	mat_W = W0

	while(couter < max_n):
		# Find our estimation of the unmixed sounds
		mat_Y = mat_W.dot(X)

		# Construct Z
		mat_Z = 1/(1+(np.exp(-1*mat_Y)))

		# Find dW, amount we want to change by
		dW = alpha * ((np.eye(num_signals) + (1-2*mat_Z).dot(np.transpose(mat_Y))/X.shape[1]).dot(mat_W))

		# Increment W
		mat_W += dW

		# Count this iteration
		couter += 1

	return [mat_W, mat_W.dot(X), couter]

# Plots 2 signals on a new figure
def plot2Signals(sig1, label1, sig2, label2):
	plt.figure()
	plt.plot(sig1, 'r-', label=label1)
	plt.plot(sig2, 'g-', label=label2)
	plt.legend(loc="best")
	plt.xlabel('Sample Number')
	plt.ylabel('Sound Output')
	plt.title('Before and After ICA')

# Creates an nxn matrix of 2-norms between two sets of signals
# The lowest in each row should coorespond to signals that match
# Returns the matrix, and an array cooresponding to the lowest in each row
def findNearestSignal(sigs1, sigs2):
	dists = np.zeros((sigs1.shape[0], sigs1.shape[0]))

	for i in range(sigs1.shape[0]):
		for j in range(sigs1.shape[0]):
			# dists[i,j] = np.linalg.norm((sigs1[i,:] - sigs2[j,:]), 2)
			dists[i,j] = abs(np.correlate(sigs1[i,:], sigs2[j,:]))

	# return [dists, np.argmin(dists, axis=1)]
	return [dists, np.argmax(dists, axis=1)]

# Creates 1 figure for each signal, before and after ICA
def plotCoorespondingSignals(sigs1, label1, sigs2, label2, t0, t1):
	[dists, locs] = findNearestSignal(sigs1, sigs2)

	for i in range(locs.shape[0]):
		plot2Signals(sigs1[i,t0:t1], label1, sigs2[locs[i],t0:t1], label2)

# Tests 1 run of ICA for given values
# All same inputs as ICA, plus the actual sounds U
# Returns unmixed sounds Y, and norms
def testICA(mat_U,W0,mat_X,alpha,max_n,epsilon):
	[W,Y,iterations] = performICA(W0, mat_X, alpha, max_n, epsilon)

	# Normalize the output
	Y = normalizeMatrix(Y)

	[norms, locs] = findNearestSignal(Y, mat_U)

	# return [Y, np.amin(norms,axis=1)]
	return [Y, np.amax(norms,axis=1)]

# Exact same as testICA, but without epsilon
def testICAforN(mat_U,W0,mat_X,alpha,max_n):
	[W,Y,iterations] = performICAforN(W0, mat_X, alpha, max_n)

	# Normalize the output
	Y = normalizeMatrix(Y)

	[norms, locs] = findNearestSignal(Y, mat_U)

	# return [Y, np.amin(norms,axis=1)]
	return [Y, np.amax(norms,axis=1)]
