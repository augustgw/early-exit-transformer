from conf import *
import torchaudio.transforms as T
import torch.nn.functional as F

from typing import List 
from torchaudio.models.decoder import ctc_decoder
import sys
import re
import os
from models.model.early_exit import Early_encoder
#from inference import model
from util.data_loader import text_transform

#for bigger LM
LM_WEIGHT = 1.0#3.23
WORD_SCORE = -0.26
N_BEST = 1

'''
#for smaller LM
LM_WEIGHT = 10.0
WORD_SCORE = -0.26
N_BEST = 1
'''
if bpe_flag== True:
    decoder=[]
    for w_ins in [-1,-1,-1,-2,-2.3, -2.3]:
        decoder += [ctc_decoder(lexicon=lexicon,
                                tokens=tokens,
                                nbest=N_BEST,
                                log_add=True,
                                beam_size=100,
                                word_score=w_ins,
                                lm_weight=LM_WEIGHT,
                                blank_token="@",
                                unk_word="<unk>",
                                sil_token="<pad>" )]   
else:
    beam_search_decoder = ctc_decoder(
        lexicon=lexicon,
        tokens=tokens,
        nbest=N_BEST,
        log_add=True,
        beam_size=1500,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE
)                     
#lm="lm.bin"
#    lm="4gram_small.arpa.lm",



class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank
        def forward(self, emission: torch.Tensor) -> List[str]:
            """Given a sequence emission over labels, get the best path
               Args:
                emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
                Returns
                List[str]: The resulting transcript      
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return indices

greedy_decoder = GreedyCTCDecoder()                                                                                   


def beam_predict(model, input_sequence, words=None, vocab_size = dec_voc_size, max_length=300, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token = trg_pad_idx, weight_ctc = 0.5):

    emission = model.ctc_encoder(input_sequence)

    beam_search_result = beam_search_decoder(emission.cpu())
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    return(beam_search_transcript)

def ctc_predict(emission, index=5):
    beam_search_result = decoder[index](emission.cpu())
    beam_search_transcript = []
    for s_ in beam_search_result:
        beam_search_transcript = beam_search_transcript + [" ".join(s_[0].words).strip() ]
    return(beam_search_transcript)



def get_trellis(emission, tokens, blank_id=0):
    
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(device)
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

from dataclasses import dataclass
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    ###t_start = torch.argmax(trellis[:, j]).item()
    t_start=trellis.size(0)-1
    path = []
    prob = 0
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        #prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        prob = prob + emission[t - 1, tokens[j - 1] if changed > stayed else 0].item()        
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))
    
        # 3. Update the token

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    if j >  0:
        #raise ValueError("Failed to align")
        print(t,j,"Failed to align")        
    return path[::-1]

def avg_models(model, path, init, end):
    nepoch=init

    best_model=path+'mod{:03d}-transformer'.format(nepoch)
    model.load_state_dict(torch.load(best_model,map_location=device))
    m1=model.state_dict()
    nc = 1

    for nepoch in range(nepoch+1,end+1):
        best_model=path+'/mod{:03d}-transformer'.format(nepoch)            
        if os.path.exists(best_model):
            print("Averaging with:", best_model)
            model.load_state_dict(torch.load(best_model,map_location=torch.device(device)))
            m2=model.state_dict()
            for key in m2:
                m1[key] = m2[key]+m1[key]
            nc = nc +1
            del m2
    
    for key in m1:
        m1[key] = m1[key] / nc
        
    model.load_state_dict(m1)
    del m1
    return model


