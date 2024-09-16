import torch , torchaudio, librosa, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, WhisperModel
import whisper

from modules import load_audio, MosPredictor, denorm
import numpy as np
import pandas as pd
import os


#check if the bathc is the correct as expected the train function(todo ) checked 
class MeuAudio(Dataset):
    def __init__(self, mos_list, whisper_list):
        self.mos_data = pd.read_csv(mos_list)
        self.whisper_data = pd.read_csv(whisper_list)

    def __getitem__(self, idx):
        row = self.mos_data.iloc[idx]
        wave_path = row['file_path']
        
        score_quality = torch.tensor(float(row['mos']))

        wav = torchaudio.load(wave_path)[0]
       
        lps = torch.from_numpy(np.expand_dims(np.abs(librosa.stft(wav[0].detach().numpy(), n_fft=512, hop_length=256, win_length=512)).T, axis=0))

        whisper_path = self.whisper_data.iloc[idx]['whisper_path']
        whisper = np.load(whisper_path)
        
        return wav, lps , whisper , score_quality
    
    def __len__(self):
        return len(self.mos_data)
    
    


def len_data(dataloader):
    for i, (wav, lps, whisper, label_quality) in enumerate(dataloader):
        print(f"Exemplo {i+1}")
        print(f"Dimensões do wav: {wav.shape}")  # Exibe as dimensões do tensor wav
        print(f"Dimensões do lps: {lps.shape}")  # Exibe as dimensões do tensor lps

        if isinstance(whisper, np.ndarray):
            print(f"Dimensões do whisper (NumPy array): {whisper.shape}")  # Exibe as dimensões do NumPy array whisper
        else:
            data = np.load(whisper)
            print("Tipo desconhecido para whisper:", data.shape)

        print(f"Label de qualidade: {label_quality.shape}")  # Exibe as dimensões do tensor de label (MOS)
        print('-' * 50)  # Linha separadora para clareza

        if i == 10:
            break


#### atenção tem que ter .npy
def freeze_intel_layer(self):
    for param in self.att_output_layer_intell.parameters():
        param.requires_grad = False
    
    for param in self.output_layer_intell.parameters():
        param.requires_grad = False
    
    for param in self.intellaverage_score.parameters():
        param.requires_grad = False

def main():
    parser = argparse.ArgumentParser('Fine-tuning do modelo MOSA NET+')
    parser.add_argument('--csv_path', type=str, help='Caminho para o arquivo CSV')
    args = parser.parse_args()
    whisper_path = 'TreinoWhisper.csv'
    #audio_files, labels_quality = load_data(args.csv_path)
    train_data = 'Treino.csv'

    train_dataset = MeuAudio(mos_list=train_data ,whisper_list=whisper_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    #val_dataset e val_dataloader
    #test_dataset e test_dataloader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MosPredictor().to(device)
   # model.freeze_intel_layer() #supostamente isso congela as camadas de inteligibilidade
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    len_data(train_dataloader)
    
    
    
   # train(model, dataloader, optimizer, criterion, device, num_epochs=10)
   # eval(model, dataloader, criterion, device)

if __name__ == "__main__":
    main()

    
