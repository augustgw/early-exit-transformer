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
from models.model.decoder import Decoder
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

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers, n_decoder_layers, features_length, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder_1 = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.positional_encoder_2 = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)        
        self.emb = nn.Embedding(dec_voc_size, d_model)
        self.linear_1=nn.Linear(d_model, dec_voc_size)
        self.linear_2=nn.Linear(d_model, dec_voc_size)
        
        
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=dim_feed_forward,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_encoder_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=dim_feed_forward,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_decoder_layers,
                               device=device)
        
    def _encoder_(self, src: Tensor) -> Tensor:
        src = self.conv_subsample(src)
        src = self.positional_encoder_1(src.permute(0,2,1))
        src_pad_mask = None
        enc_out=self.encoder(src, src_pad_mask)
        return enc_out        

    def ctc_encoder(self, src: Tensor) -> Tensor:
        src = self.conv_subsample(src)
        src = self.positional_encoder_1(src.permute(0,2,1))
        src_pad_mask = None
        enc_out=self.encoder(src, src_pad_mask)
        enc_out = self.linear_1(enc_out) 
        enc_out = torch.nn.functional.log_softmax(enc_out,dim=2)
        return enc_out        

    def _decoder_(self, trg: Tensor, enc: Tensor, src_trg_mask: Optional[Tensor] = None) -> Tensor:

        src_pad_mask = None
        #print("MASK:",src_trg_mask)
        trg=self.emb(trg)

        trg=self.positional_encoder_2(trg)
        #src_trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_no_peak_mask(trg, trg)
        src_trg_mask = self.create_tgt_mask(trg.size(1)).to(self.device)
        output = self.decoder(trg, enc, src_trg_mask, src_pad_mask)
        output = self.linear_2(output)        
        output = torch.nn.functional.log_softmax(output,dim=2)

        return output
    
    def forward(self, src, trg):

        #Encoder
        src = self.conv_subsample(src)
        src = self.positional_encoder_1(src.permute(0,2,1))

        src_pad_mask = None 
        enc = self.encoder(src, src_pad_mask)                                        
        
        #Decoder


        src_trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_no_peak_mask(trg, trg)
        #src_trg_mask = self.create_tgt_mask(trg.size(1)).to(self.device)        
        
        #print("MASK:",src_trg_mask)
        trg=self.emb(trg)
        trg=self.positional_encoder_2(trg)

        output = self.decoder(trg, enc, src_trg_mask, src_pad_mask)        
        output = self.linear_2(output)
        output = torch.nn.functional.log_softmax(output,dim=2)
        
        #for ctc loss
        enc = self.linear_1(enc) 
        enc = torch.nn.functional.log_softmax(enc,dim=2)

        return output, enc

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def create_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
 
        return torch.tril(torch.full((sz, sz), bool(True)), diagonal=0).unsqueeze(0).unsqueeze(1)

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)
        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)

        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)

        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q

        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask

    


    
class CTC_Self_Attention(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, dec_voc_size, d_model, n_head, max_len,  dim_feed_forward, n_encoder_layers, features_length, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.linear=nn.Linear(d_model, dec_voc_size)
        
        
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=dim_feed_forward,
                               enc_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_encoder_layers,
                               device=device)

    def forward(self, src):

        #Encoder
        src = self.conv_subsample(src)
        
        src = self.positional_encoder(src.permute(0,2,1))
        
        src_pad_mask = None 
        enc = self.encoder(src, src_pad_mask)
        
        enc = self.linear(enc) 
        enc = torch.nn.functional.log_softmax(enc,dim=2)

        return enc

    
