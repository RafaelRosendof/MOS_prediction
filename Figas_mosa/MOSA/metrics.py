import os 
import csv
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
import numpy as np
#sklearn e outras bibliotecas com as métricas

def calcular_metricas(y_true, y_pred):
    """Calcula as métricas MSE, LCC, SRCC, KTAU entre dois vetores."""
    mse = mean_squared_error(y_true, y_pred)
    lcc, _ = pearsonr(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)
    ktau, _ = kendalltau(y_true, y_pred)
    
    return {
        'MSE': mse,
        'LCC': lcc,
        'SRCC': srcc,
        'KTAU': ktau
    }

def leitura(file1, file2, outFile):
    """Lê dois arquivos CSV, calcula as métricas e escreve os resultados em um arquivo de saída."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        read1 = list(csv.reader(f1))
        read2 = list(csv.reader(f2))

        y_true = np.array([float(row[1]) for row in read1])
        y_pred = np.array([float(row[1]) for row in read2])

        # Calcular as métricas
        metricas = calcular_metricas(y_true, y_pred)

        # Escrever os resultados em um arquivo CSV
        with open(outFile, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['Métrica', 'Valor'])
            for chave, valor in metricas.items():
                writer.writerow([chave, valor])

        print(f"Métricas salvas em {outFile}")
        
        
        
def main():
    print("Comparing metrics")

    parser = argparse.ArgumentParser(description='Comparing metrics')

    parser.add_argument("--i1" , type=str , help="Input file 1")
    parser.add_argument("--i2" , type=str , help="Input file 2")
    parser.add_argument("--o" , type=str , help="Output file")

    args = parser.parse_args()

    leitura(args.i1 , args.i2 , args.o)

    print("Metrics done successfully")

if __name__ == "__main__":
    main()

