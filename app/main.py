import csv

def main():
    print('''Welcome to the main script, this code implements the logic and calls other methods that implement the specific architecture''')

    while True: 
        csv_data=getCSVData()
        if csv_data is not None:
            print('I have a csv')
        else: 
            print('no CSV provided')


def getCSVData():
    """This function, given a path, finds the csv data"""
    path_csv=input('> Path to the csv file?')
    with open(path_csv, newline='') as csvfile:
        csv_reader=csv.reader(csvfile)
        header=next(csv_reader)
        num_cols=len(header)
        num_rows=sum(1 for row in csv_reader)
        print(f"Found: {num_cols} columns and {num_rows} rows in csv.")
    return path_csv



if __name__=='__main__':
    main()
