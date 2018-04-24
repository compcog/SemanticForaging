import sys
import foragingModel

def main():
    data_file = './Data/data-psyrev.txt'
    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
        print('Using data file "' + data_file + '"...')
    else:
        print('No data file parameter provided, using "' + data_file + '"...')

    foragingModel.modelFits(data_file)

if __name__ == "__main__":
    main()
