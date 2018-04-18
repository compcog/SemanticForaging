Organizational Structure:
./
	foragingModel.py
	/Data
		inputFile.whatever
		/corpus
			corrections.txt
			frequencies.csv
			similaritylabels.csv
			similaritymatrix.csv
		/results
		
To get parameter fits for each subject, import the foraging model python file:

	import foragingModel

Then call the model fitting method, passing the path to the subject data file:

	foragingModel.modelFits('Data/data-psyrev.txt')

The model will perform some low-level text pre-processing, then find the best fitting parameters for each subject. Once parameters have been fit to each model, the method will save the data to:

	/Data/results/fullfits.csv
	/Data/results/fullmetrics.csv
	
where fullfits.csv records the best fitting parameters for each participant, and where fullmetrics.csv records entries, word frequencies, word similarities, and patch switches for each participant.