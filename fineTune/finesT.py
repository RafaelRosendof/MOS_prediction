import torch , torchaudio, librosa, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, WhisperModel
import whisper

from mosaNET import MosPredictor , denorm , frame_score
import numpy as np
import pandas as pd
from tqdm import tqdm
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



def freeze_intel_layer(self):
    # Congelar camadas relacionadas à inteligibilidade
    for param in self.att_output_layer_intell.parameters():
        param.requires_grad = False
    
    for param in self.output_layer_intell.parameters():
        param.requires_grad = False
    
    for param in self.intellaverage_score.parameters():
        param.requires_grad = False


        
def train(ckpt_path, model, trainloader, optimizer, criterion, device, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = MosPredictor()
    net = net.to(device)
    
    net.load_state_dict(torch.load(ckpt_path))

    
    PREV_VAL_LOSS = 999999999
    orig_patience=5
    patience = orig_patience

    for epoch in range(num_epochs):
        STEPS = 0

        net.train()
        run_loss = 0.0
        
        for i ,data in enumerate(tqdm(trainloader), 0):
            inputs , lps , whisper , label_quality = data
            inputs = inputs.to(device)
            lps = lps.to(device)
            whisper = whisper.to(device)
            label_quality = label_quality.to(device)

            optimizer.zero_grad()
            output_quality , frame_quality = net(inputs, lps, whisper)
            
            label_frame_quality = frame_score(label_quality, frame_quality)

            loss_frame_quality = criterion(frame_quality, label_frame_quality)

            loss_quality = criterion(output_quality.squeeze(1) , label_quality)
            
            loss = loss_quality + loss_frame_quality
            
            loss.backward()
            optimizer.step()
            STEPS += 1
            run_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {run_loss/STEPS}")
        print(f"Epoch {epoch+1} - Loss Quality: {loss_quality.item()}")
    
        
        epoch_val_loss = 0.0
        net.eval()
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 
    print("Treinamento concluído")
     

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
    model.freeze_intel_layer() #supostamente isso congela as camadas de inteligibilidade
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    len_data(train_dataloader)
    
    
    
   # train(model, dataloader, optimizer, criterion, device, num_epochs=10)
   # eval(model, dataloader, criterion, device)

if __name__ == "__main__":
    main()

    
