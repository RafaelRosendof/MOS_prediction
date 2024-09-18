import pandas as pd
import argparse 

def read(path, out_path):
    # Lê o arquivo CSV
    df = pd.read_csv(path)

    # Abre o arquivo de saída no modo de escrita
    with open(out_path, 'w') as f:
        for index, row in df.iterrows():
            # Escreve os dados no formato: mos_quality,mos_intelligibility,caminho_do_arquivo
            f.write(f"{row['mos_quality']},{row['mos_intelligibility']},{row['file_path']}\n")

    print(f"Arquivo convertido com sucesso! Salvo em: {out_path}")
    
    
def main():
    
    parser = argparse.ArgumentParser(description='Convert csv to txt')
    parser.add_argument('--path', type=str, help='path to csv file')
    parser.add_argument('--o', type=str, help='path to output txt file')

    args = parser.parse_args()

    read(args.path , args.o)

    print("Done")
