"""
@author : Hyunwoong 
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
@author : Daniele Falavigna
@when : 2023-03-10
"""
import sys
import torch
from torch import nn
from torch import Tensor
from typing import Optional, Any, Union, Callable   
from torchaudio.models.conformer import Conformer

from models.model.encoder import Encoder
from models.embedding.positional_encoding import PositionalEncoding

torch.set_printoptions(profile='full')        
class Conv1dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs               

class Conv2dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU()
        )

    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs

    
class Early_transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, n_enc_replay, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers, n_decoder_layers, features_length, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.n_enc_replay=n_enc_replay
        self.device = device
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.conv2_subsample = Conv2dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder_1 = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.positional_encoder_2 = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)        
        self.emb = nn.Embedding(dec_voc_size, d_model)
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-5)

        self.linears_1=nn.ModuleList([nn.Linear(d_model, dec_voc_size) for _ in range(self.n_enc_replay)])
        self.linears_2=nn.ModuleList([nn.Linear(d_model, dec_voc_size) for _ in range(self.n_enc_replay)])
        
        self.encoders = nn.ModuleList([Encoder(d_model=d_model,
                            n_head=n_head,
                            max_len=max_len,
                            ffn_hidden=dim_feed_forward,
                            enc_voc_size=enc_voc_size,
                            drop_prob=drop_prob,
                            n_layers=n_encoder_layers,
                            device=device) for _ in range(self.n_enc_replay)])

        self.decoders = nn.ModuleList([nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model,
                            nhead=n_head,
                            dim_feedforward=dim_feed_forward,
                            dropout=drop_prob,
                            batch_first= "True",
                            norm_first = "True"),
                            n_decoder_layers, self.layer_norm) for _ in range(self.n_enc_replay)])
    
    def forward(self, src, trg):

        #Encoder
        src = self.conv_subsample(src)
        src = self.positional_encoder_1(src.permute(0,2,1))
        tgt_mask = self.create_tgt_mask(trg.size(1)).to(self.device)        
        tgt_key_padding_mask = self.create_pad_mask(trg, self.trg_pad_idx).to(self.device)
        trg=self.emb(trg)
        trg=self.positional_encoder_2(trg)

        src_pad_mask = None
        enc = src
        output, enc_out= [], []
        for linear_1, linear_2, encoder, decoder  in zip(self.linears_1, self.linears_2, self.encoders, self.decoders):
            enc = encoder(enc, src_pad_mask)                                        
            
            #Decoder
            out = decoder(trg,enc,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask) 
            out = linear_2(out)
            out = torch.nn.functional.log_softmax(out,dim=2)
            output += [out.unsqueeze(0)]#output.append(out.unsqueeze(0))
            #for ctc loss
            out = linear_1(enc) 
            out = torch.nn.functional.log_softmax(out,dim=2)
            enc_out += [out.unsqueeze(0)]#output.append(out.unsqueeze(0))
        output=torch.cat(output)
        enc_out=torch.cat(enc_out)
        
        return output, enc_out


    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def create_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class Early_encoder(nn.Module):

    def __init__(self, src_pad_idx, n_enc_replay, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers,  features_length, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.n_enc_replay=n_enc_replay
        self.device = device
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)

        self.linears=nn.ModuleList([nn.Linear(d_model, dec_voc_size) for _ in range(self.n_enc_replay)])
        
        self.encoders = nn.ModuleList([Encoder(d_model=d_model,
                            n_head=n_head,
                            max_len=max_len,
                            ffn_hidden=dim_feed_forward,
                            enc_voc_size=enc_voc_size,
                            drop_prob=drop_prob,
                            n_layers=n_encoder_layers,
                            device=device) for _ in range(self.n_enc_replay)])

    def forward(self, src):

        #Encoder
        src = self.conv_subsample(src)
        src = self.positional_encoder(src.permute(0,2,1))
        src_pad_mask = None
        enc = src
        enc_out= []
        for linear, encoder  in zip(self.linears, self.encoders):
            enc = encoder(enc, src_pad_mask)                                        
            
            #for ctc loss
            out = linear(enc) 
            out = torch.nn.functional.log_softmax(out,dim=2)
            enc_out += [out.unsqueeze(0)]#output.append(out.unsqueeze(0))
        enc_out=torch.cat(enc_out)
        
        return enc_out

class Early_conformer(nn.Module):

    def __init__(self, src_pad_idx, n_enc_replay, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers,  features_length, drop_prob, depthwise_kernel_size, device):
        super().__init__()
        self.input_dim=d_model
        self.num_heads=n_head
        self.ffn_dim=dim_feed_forward
        self.num_layers=n_encoder_layers
        self.depthwise_conv_kernel_size=depthwise_kernel_size
        self.n_enc_replay=n_enc_replay
        self.dropout=drop_prob
        self.device=device
        
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.linears=nn.ModuleList([nn.Linear(d_model, dec_voc_size) for _ in range(self.n_enc_replay)])
        self.conformer=nn.ModuleList([Conformer(input_dim=self.input_dim, num_heads=self.num_heads, ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, dropout=self.dropout) for _ in range(self.n_enc_replay)])

    def forward(self, src, lengths):

        #convolution
        src = self.conv_subsample(src)
        src = self.positional_encoder(src.permute(0,2,1))

        length=torch.clamp(lengths/4,max=src.size(1)).to(torch.int).to(self.device)
        enc_out= []
        enc=src
        for linear, layer  in zip(self.linears, self.conformer):
            enc, _ = layer(enc, length)
            #for ctc loss
            out = linear(enc) 
            out = torch.nn.functional.log_softmax(out,dim=2)
            enc_out += [out.unsqueeze(0)]#output.append(out.unsqueeze(0))
        enc_out=torch.cat(enc_out)
        
        return enc_out
        
class my_conformer(nn.Module):

    def __init__(self, src_pad_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers,  features_length, drop_prob, depthwise_kernel_size, device):
        super().__init__()
        self.input_dim=d_model
        self.num_heads=n_head
        self.ffn_dim=dim_feed_forward
        self.num_layers=n_encoder_layers
        self.depthwise_conv_kernel_size=depthwise_kernel_size
        self.dropout=drop_prob
        self.device=device
        
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.linear=nn.Linear(d_model, dec_voc_size)
        self.conformer=Conformer(input_dim=self.input_dim, num_heads=self.num_heads, ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, dropout=self.dropout)

    def forward(self, src):

        #convolution
        src = self.conv_subsample(src)
        src = self.positional_encoder(src.permute(0,2,1))
        length = torch.full(size=(src.size(0),), fill_value = src.size(1), dtype=torch.long).to(self.device)

        enc, _ = self.conformer(src, length)
        out = self.linear(enc) 
        out = torch.nn.functional.log_softmax(out,dim=2)

        return out.unsqueeze(0)
        
