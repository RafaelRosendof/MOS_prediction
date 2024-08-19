import csv
import os
import argparse

def achou(inp1 , inp2 , out):
    with open(inp2, 'r') as f2:
        reader2 = csv.reader(f2)
        set_inp2 = {row[0] for row in reader2}  

    
    with open(inp1, 'r') as f1, open(out, 'w', newline='') as fout:
        reader1 = csv.reader(f1)
        writer = csv.writer(fout)
        for row in reader1:
            if row[0] in set_inp2: 
                writer.writerow(row)
        
            
            
def main():
    print("Iniciando a busca")
    parser = argparse.ArgumentParser(description='Procura linhas de um arquivo em outro arquivo')
    parser.add_argument('--i1' , type=str , help='Arquivo 1')
    parser.add_argument('--i2' , type=str , help='Arquivo 2')
    parser.add_argument('--o' , type=str , help='Arquivo de saida')

    args = parser.parse_args()

    achou(args.i1 , args.i2 , args.o)

if __name__ == "__main__":
    main()