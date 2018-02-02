#documents before words_to_probs
# {
# ('doc1',('word1','word2', ... , 'wordN')
# }

# prob_dict before words_to_probs
# {
# word1: (p(class1), p(class2), p(class3), p(class4)),
# word2: (p(class1), p(class2), p(class3), p(class4)),
# ...
# wordN: (p(class1), p(class2), p(class3), p(class4))
# }

# priors_dict
# {
# class1: prior(class1),
# class2: prior(class2),
# class3: prior(class3),
# class4: prior(class4)
# }

# replaces each word of each document with a tuple
# of the form (word, x) where x is a list of conditional probabilities over each class
def words_to_probs(documents, prob_dict):
	# convert words to their conditional probabilities for each class
	return documents.map(lambda doc: (
				doc[0],
				[prob_dict[word] for word in doc[1] if word in prob_dict] if (type(doc[1])==tuple or type(doc[1])==list) else prob_dict[doc[1]]
			))

# called on the output of words_to_probs, this aggregates the conditional probabilities
# for each word and the priors over the relevant classes, leaving one list of
# conditional probabilities for each document
def docs_to_probs(documents, priors_dict):
	return documents.map(lambda document: total_prob(document, priors_dict))

# main implementation for docs_to_probs
def total_prob(document, priors_dict):
	p = [priors_dict['CCAT'], priors_dict['ECAT'], priors_dict['GCAT'], priors_dict['MCAT']]  # initialize probability as priors
	if type(document[1]) == list:
		for word_probs in document[1]:	# document[0] is name, [1] is word probs
			p = [sum(x) for x in zip(p, word_probs)] # add log probabilities element-wise to p
	elif type(document[1]) == tuple: #if it's a tuple then there's only one, so no for loop
		p = [sum(x) for x in zip(p, document[1])]
	return (document[0], p)

# given an rdd of lists of condtional probabilities (each corresponding to a document)
# class_preds returns an rdd of indices of the greatest probability for each.
def class_preds(documents):
	return documents.map(lambda document: (document[0], document[1].index(max(document[1]))))
	# correspondances: 0 - CCAT, 1 - ECAT, 2 - GCAT, 3 - MCAT

# used to evaluate the accuracy of the output of class_preds (as the culmination of previous processing)
# against the known class labels
def accuracy(predictions, labels):
	labels = labels_to_indexes(labels)
	num_examples = predictions.count()
	if(num_examples != labels.count()):
		print("Error in 'accuracy' method:  must have same # of predictions and labels")
		return -1
	combined = predictions.join(labels)
	correct = combined.map(lambda x: int(x[1][0] in x[1][1]))
	num_correct = correct.reduce(lambda x, y: x+y)
	return (float)(num_correct/num_examples)

# used in accuracy function to convert labels to indices 
# which correspond with the output of the class_preds function
def labels_to_indexes(labels):
	mapping = {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3}
	return labels.map(lambda label: (label[0], [mapping[x] for x in label[1]] if type(label[1])==tuple else [mapping[label[1]]]))
