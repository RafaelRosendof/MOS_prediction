import csv
import argparse

def remove(file , out , strFigas):
    with open(file , 'r') as f:
        lines = f.readlines()

    with open(out , 'w') as f:
        for line in lines:
            mod = line.split(strFigas, 1)[-1]
            f.write(mod)


def main():
    print("Iniciando")

    parser = argparse.ArgumentParser(description='Remove até um local')
    parser.add_argument("--i" , type=str , help="Input file")
    parser.add_argument("--o" , type=str , help="Output file")
    parser.add_argument("--str" , type=str , help="String to remove")

    args = parser.parse_args()

    remove(args.i , args.o , args.str)

    print("\n\n deu tudo certo")

if __name__ == "__main__":
    main()
        