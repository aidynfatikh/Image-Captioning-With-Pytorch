import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass


class EncoderCNN(nn.Module):
    def __init__(self, config):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.requires_grad_(False)  # freeze layers
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove last layer
        self.proj = nn.Linear(resnet.fc.in_features, config.n_embd)  

    def forward(self, images):
        features = self.resnet(images).flatten(start_dim=1)  # extract & flatten features
        return self.proj(features)  # project to embedding size
    
class DecoderRNN(nn.Module):
    def __init__(self, config):
        super(DecoderRNN, self).__init__()   
        self.n_embd = config.n_embd 
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers 

        self.word_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.lstm = nn.LSTM(self.n_embd, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True,
                            dropout=0.2)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1] # we dont use last word for prediction
        captions = self.word_embedding(captions) # construct word embeddings (batch_size, seq_length-1, n_embd)
        captions = torch.cat((features.unsqueeze(1), captions), 1) # stick the features as the first input (batch_size, seq_length, n_embd)

        output, _ = self.lstm(captions) # (batch_size, seq_length, hidden_size)
        output = self.fc(output) # (batch_size, seq_length, vocab_size)
        return output[:, 1:, :] # return last word
    
@dataclass
class ITTConfig:
    batch_size = 64
    hidden_size = 256
    num_layers = 2
    n_embd = 256
    pad_token = 0
    vocab_size = None # is set in train.py

class ITT(nn.Module):
    def __init__(self, config):
        super(ITT, self).__init__()
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.encoder = EncoderCNN(config)
        self.decoder = DecoderRNN(config)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=config.pad_token)

    def forward(self, image, caption):
        features = self.encoder(image)
        features = self.decoder(features, caption)
        loss = self.criterion(features.reshape(-1, self.vocab_size), caption[:, 1:].reshape(-1, ))
        
        return loss, features

    def sample(self, image, vocab, max_len=20):
        output = []
        features = self.encoder(image)
        features = features.unsqueeze(0) # unsqueeze the batch dimension

        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=features.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size, device=features.device)

        for _ in range(max_len):
            x, (h, c) = self.decoder.lstm(features, (h, c))
            x = self.decoder.fc(x)
            x = x.squeeze(1) # squeeze the sequence dimension
            predict = x.argmax(dim=1)

            if predict.item() == vocab.stoi["<end>"]: # end sampling after end token
                break

            output.append(predict.item())
            features = self.decoder.word_embedding(predict.unsqueeze(0))
        
        output = [vocab.itos[i] for i in output] # convert to strings
        return output
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} 

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        #print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} params")
        #print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params} params")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

