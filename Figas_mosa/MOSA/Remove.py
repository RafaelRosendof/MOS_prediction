import pandas as pd
import os 
import argparse


def removeCol(file , col):
   df = pd.read_csv(file)
   
   if col in df.columns:
       df.drop(columns=[col] , inplace=True)
       df.to_csv(file , index=False)
    
   else:
         print("Column not found")
   

def main():
    print("Removing files")
    parser = argparse.ArgumentParser(description='Remove Columns')
    parser.add_argument("--i" , type=str, help="Input file")
    parser.add_argument("--col" , type=str , help="Column to remove")

    args = parser.parse_args()

    removeCol(args.i , args.col)

if __name__ == "__main__":
    main()
