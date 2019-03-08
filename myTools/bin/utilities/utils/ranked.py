import numpy as np

def rank5_accuracy(preds, labels):
	# initialzie the rank-1 and rank-5 accuracies
	rank1 = 0
	rank5 = 0

	# loop over the predictions and ground-truth labels
	for (p, gt) in zip(preds, labels):
		# sort the probabilites by their index in descending
		# order to that the more confident guesses are at the
		# front of the list
		p = np.argsort(p)[::-1]

		# check if the ground-truth label is in the top-5
		# predictions
		if gt in p[:5]:
			rank5 += 1

		# check if the ground-truth label is the #1 prediction
		if gt == p[0]:
			rank1 += 1

	# compute the finak rank-1 and rank-5 accuracies
	rank1 /= float(len(labels))
	rank5 /= float(len(labels))
	
	# return a tuple of the rank-1 and rank-5 accuracies
	return(rank1, rank5)
