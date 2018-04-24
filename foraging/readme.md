## How to use

On the command line, run `python run.py <input-file-name>`. If you don't specify an input file, the tool will use the example input file in `Data/data-psyrev.txt`. The tool will write the results of the fit into the `Data/results` folder.

If there are misspellings of animals or unknown animals, you can either fix these in the data file directly, or add lines to the `Data/corpus/corrections.txt` file. If a participant response is not found in the dictionary or the corrections file, the fitting algorithm will remove that response from the list.

**Dependencies**

- python 3
- pandas
- numpy

## Details

Organizational Structure:

```
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
```

To get parameter fits for each subject, import the foraging model python file:

`import foragingModel`

Then call the model fitting method, passing the path to the subject data file:

`foragingModel.modelFits('Data/data-psyrev.txt')`

The model will perform some low-level text pre-processing, then find the best fitting parameters for each subject. Once parameters have been fit to each model, the method will save the data to:

```
	/Data/results/fullfits.csv
	/Data/results/fullmetrics.csv
```

where `fullfits.csv` records the best fitting parameters for each participant, and where `fullmetrics.csv` records entries, word frequencies, word similarities, and patch switches for each participant.
