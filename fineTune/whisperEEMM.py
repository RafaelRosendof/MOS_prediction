import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import whisper
import pdb
import torch
import argparse
import numpy as np 
from transformers import AutoFeatureExtractor, WhisperModel
import torch.nn as nn

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def extract(input_list, filename, directory, mode):
    device = torch.device("cuda")
    model_asli = WhisperModel.from_pretrained("openai/whisper-large-v3")   
    model = model_asli.to(device)  

    list_new=[]   
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")       
    for i in range(len(input_list)):   
        
        if mode == 'train':
           Asessment_filepath=input_list[i].split(',')
           wavefile = Asessment_filepath[2]
        else:
           Asessment_filepath=input_list[i]
           wavefile = Asessment_filepath          
        
        path_name = wavefile
        S=path_name.split('/')
        wave_name=S[-1]
        name = wave_name[:-4] 
        new_name =  name +'.npy' 
        #caso queira salvar como .pth
        # new_name = name + '.pth'
               
        cached_path = os.path.join(directory,new_name)
        
        audio = whisper.load_audio(wavefile)
        inputs = feature_extractor(audio, return_tensors="pt")
        input_features = inputs.input_features
        input_features = input_features.to(device)  

        decoder_input_ids = torch.tensor([[1, 1]]) * model_asli.config.decoder_start_token_id
        decoder_input_ids =  decoder_input_ids.to(device)
        '''
        opção 1
        
        model_asli.config.output_hidden_states = True
        model_asli.config.output_hidden_states = True

        outputs = model(input_features , decoder_input_ids=decoder_input_ids)

        intermediate_layers = outputs.encoder_hidden_states[5] #pegar a camada 6

        feat = intermediate_layers.detach().to("cpu").numpy()
        np.save(cached_path,feat)

        
        opção 3
        # Obter todos os estados ocultos intermediários
        outputs = model(input_features, decoder_input_ids=decoder_input_ids)

        # `encoder_hidden_states` conterá as saídas de todas as camadas
        hidden_states = outputs.encoder_hidden_states

        # Por exemplo, acessando a camada intermediária 6
        intermediate_layer = hidden_states[6]  # ou qualquer camada 
        feat = intermediate_layer.detach().to("cpu").numpy()
        
        '''

        #caso queira pegar camadas intermediárias 
    
       #Apaga essa linha  
        last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state 

        #feat = intermediate_layers.detach().to("cpu").numpy()
        feat = last_hidden_state.detach().to("cpu").numpy()
        np.save(cached_path,feat)        
        #torch.save(feat , cached_path) #para salvar via .pth
        info = str(cached_path)
        list_new.append(info)
    
    with open(filename+'.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item) 
          
          
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')     
    parser.add_argument('--filename', type=str, default='Whisperfeat_Train_VoiceMOS_2023')       
    args = parser.parse_args()
      
    print('Starting prediction')
    list_new =[]   

    if args.mode =="train":
       directory ='/MOSA-Net_Plus_Torch/Train_SSL_Feat_Whisperv3/'
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)  
       
       Input_List=ListRead('/MOSA-Net_Plus_Torch/Train_VoiceMOS_2023.txt')   
       extract(Input_List, args.filename, directory, args.mode)  
       
    else :
       directory ='/MOSA-Net_Plus_Torch/Test_SSL_Feat_Whisperv3/'    
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)  
       
       Input_List=ListRead('/MOSA-Net_Plus_Torch/Test_VoiceMOS_2023.txt')    
       extract(Input_List, args.filename, directory, args.mode)       