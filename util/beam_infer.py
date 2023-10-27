from conf import *
import torchaudio.transforms as T
import torch.nn.functional as F

from typing import List
from torchaudio.models.decoder import ctc_decoder
import sys
import re
import os
from models.model.early_exit import full_conformer
#from inference import model
from util.data_loader import text_transform
from torchaudio.models.decoder import cuda_ctc_decoder

#for bigger LM
LM_WEIGHT = 1.0#3.23#1.0#3.23
WORD_SCORE = -4#-1.0#-0.26
N_BEST = 1#500#300

'''
#for smaller LM
LM_WEIGHT = 10.0
WORD_SCORE = -0.26
N_BEST = 1
'''
if bpe_flag== True:
    '''
    beam_search_decoder = ctc_decoder(
        lexicon=lexicon,
        tokens=tokens,
        nbest=N_BEST,
        log_add=True,
        beam_size=150,
        word_score=WORD_SCORE,
        lm_weight=LM_WEIGHT,
        blank_token="@",
        unk_word="<unk>",
        sil_token="<pad>"
        )
    '''
    decoder=[]
    #for w_ins in [-1,-1,-1,-1,-1, -1]: #valori positivi aumentano le inserzioni
    for w_ins in [0,0,0,0,0,0]: #valori positivi aumentano le inserzioni        
    #for w_ins in [-0.5,0.5,0.5,0.5,0.5, 0.5]:        
        decoder += [ctc_decoder(lexicon=lexicon,
                                tokens=tokens,
                                nbest=N_BEST,
                                log_add=False,
                                beam_size=15, #0, #500,
                                word_score=w_ins,
                                lm_weight=LM_WEIGHT,
                                blank_token="@",
                                unk_word="<unk>",
                                sil_token="<pad>" )]
else:
    beam_search_decoder = ctc_decoder(
        lexicon=lexicon,
        tokens=tokens,
        nbest=1,
        log_add=True,
        beam_size=1,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE
)                     
#lm="lm.bin"
#    lm="4gram_small.arpa.lm",
cuda_decoder = cuda_ctc_decoder(tokens, nbest=1, beam_size=10, blank_skip_threshold=0.95)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
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



def ctc_predict_(emission, index=5):
    beam_search_result = decoder[index](emission.cpu())
    beam_search_transcript = []
    for s_ in beam_search_result:
        beam_search_transcript = beam_search_transcript + [" ".join(s_[0].words).strip() ]
    return(beam_search_transcript)

def ctc_cuda_predict(emission, index=5, tokens=tokens):
    #results = cuda_decoder(emission, encoder_out_lens.to(torch.int32))
    enc_len=torch.full(size=(1,), fill_value = emission.size(1), dtype=torch.int32).to(device)
    cuda_decoder = cuda_ctc_decoder(tokens, nbest=1, beam_size=10, blank_skip_threshold=0.95)
    results = cuda_decoder(emission, enc_len)

    #beam_search_transcript = bpe_model.decode(results[0][0].tokens).lower()

    return(results)

def ctc_predict(emission, index=5):
    beam_search_result = decoder[index](emission.cpu())

    beam_search_transcript = [ " ".join(beam_search_result[0][0].words).strip() ]
    nhyps=len(beam_search_result[0])
    hyp_score=torch.zeros(nhyps)
    
    for i in range(0,nhyps):
        hyp_score[i]=beam_search_result[0][i].score

    pprob=F.softmax(hyp_score,dim=0)
    return beam_search_transcript, pprob[0]

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



    
def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

def beam_search(model, encoder_output, layer_n, input_decoder, valid_length, w_lex=None, vocab_size = dec_voc_size, max_length=500, min_length=300, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token=trg_pad_idx, beam_size=5, pen_alpha=0.6, return_best_beam = True, weight_ctc = 0.5):

    beam_size = beam_size
    beam_size_count = beam_size
    pen_alpha = 1.0 #pen_alpha #increase -> decrease insertion
    vocab_size = vocab_size

    #decoder_input = input_decoder[:,0:input_decoder.size(1)-5]

    decoder_input  =torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    scores = torch.Tensor([0.]).to(device)
    #print("DECODER_INPUT:",  text_transform.int_to_text(decoder_input.squeeze(0)))
    #input_sequence = input_sequence.to(device)

    #encoder_output = model._encoder_(input_sequence, valid_length, layer_n).to(device)

    #_,emission = model(input_sequence,decoder_input)
    final_scores = []
    final_tokens = []

    for i in range(max_length):
        #decoder_input = F.pad(decoder_input, (0,10), mode='constant',value=PAD_token)
        
        if i==0:
            logits= model._decoder_(decoder_input, encoder_output, layer_n).detach()
        else:
            logits= model._decoder_(decoder_input, encoder_output.expand(beam_size_count, *encoder_output.shape[1:]), layer_n).detach()
        
        log_probs = logits[:,-1] / sequence_length_penalty(i+1, pen_alpha)
        scores = scores.unsqueeze(1) + log_probs
        scores, indices = torch.topk(scores.reshape(-1), beam_size_count)

        beam_indices  = torch.divide(indices, vocab_size, rounding_mode='floor')
        token_indices = torch.remainder(indices, vocab_size)
        next_decoder_input = []
        EOS_beams_index = []
        
        for ind, (beam_index, token_index) in enumerate(zip(beam_indices, token_indices)):
            
            prev_decoder_input = decoder_input[beam_index]
            
            if token_index == EOS_token and i>min_length:

                token_index = torch.LongTensor([token_index]).to(device)
                final_tokens.append(torch.cat([prev_decoder_input, token_index]))
                final_scores.append(scores[ind])
                beam_size_count -= 1
                
                #scores_list = scores.tolist()
                #del scores_list[ind]
                #scores = torch.tensor(scores_list, device=device)
                EOS_beams_index.append(ind)
                #print(f"Beam #{ind} reached EOS!")
                
            else:
                token_index = torch.LongTensor([token_index]).to(device)
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
        if len(EOS_beams_index) > 0:
            scores_list = scores.tolist()
            for tt in EOS_beams_index[::-1]:
                del scores_list[tt]
            scores = torch.tensor(scores_list, device=device)
            
        if len(final_scores) == beam_size:
            break
        
        decoder_input = torch.vstack(next_decoder_input)
    

    if i == (max_length -1): # We have reached max # of allowed iterations.

        for beam_unf, score_unf in zip(decoder_input,scores):
            final_tokens.append(beam_unf)
            final_scores.append(score_unf)
            del beam_unf
            del score_unf

        
        assert len(final_tokens) == beam_size and len(final_scores) == beam_size, ('Final_tokens and final_scores lists do not match beam_size size!')

        
    # If we want to return most probable predicted beam.
    #del encoder_output
    #del encoder_output_afterEOS
    del decoder_input
    del scores
    if return_best_beam:
        del encoder_output
        max_val = max(final_scores)

        return final_tokens, final_scores, final_tokens[final_scores.index(max_val)].tolist()
    else:

        s_ctc = torch.zeros(beam_size)
        #loss_ctc = torch.zeros(beam_size)        
        i=0
        ctc_input_len=torch.full(size=(emission.size(0),), fill_value = emission.size(1), dtype=torch.long)
        
        #for f_t, f_s in zip(final_tokens, final_scores):
        for f_t in final_tokens:
            #f_t=f_t[1:f_t.size(0)-1]
            #print(f_t)
            
            trellis = get_trellis(emission.squeeze(0).to(device), f_t).detach()
            path = backtrack(trellis, emission.squeeze(0), f_t)
            #print(path[0].score/len(path), len(path))
            '''
            stayed=path[0]
            
            count = 0
            s_ctc[i] = 0
            for p in path:
                #print(p.score, p.token_index, stayed.token_index)
                if p.token_index != stayed.token_index:
                    s_ctc[i] = s_ctc[i] + ( (stayed.score - pc.score) / count)
                    #print("stayed", pc.score, stayed.score, count, s_ctc[i])
                    stayed=p
                    count = 1
                    
                else:
                    count = count + 1
                pc=p
            s_ctc[i] = s_ctc[i] + ( (stayed.score - pc.score) / count)
            '''
            #print("final", pc.score, stayed.score, count, s_ctc[i])
            #plt.imshow(trellis[1:, 1:].T, origin="lower")
            #plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
            #plt.colorbar()
            #plt.show()

            ctc_target_len=f_t.size(0)
            #s_ctc[i] = ctc_loss(emission.permute(1,0,2),f_t.unsqueeze(0),ctc_input_len,torch.tensor(ctc_target_len)).to(device)#/len(f_t)
            s_ctc[i]=path[0].score/len(f_t) #len(f_t)#len(path)
            i= i+1

        
        s_pred = torch.exp(torch.tensor(final_scores))
        s_ctc = torch.exp(s_ctc)
        #print("PRED:",s_pred)
        #print("CTC:",s_ctc)
        
        #s_pred = s_pred / torch.sum(s_pred)
        #s_ctc = s_ctc / torch.sum(s_ctc)
        
        #loss_ctc=torch.exp(loss_ctc/len(path))
        
        max_=torch.max(s_pred,dim=0,keepdim=False)
        s_pred = s_pred / max_.values
        max_=torch.max(s_ctc,dim=0,keepdim=False)
        s_ctc = s_ctc / max_.values
        
        #print("PRED:",s_pred)
        #print("CTC:",s_ctc)

        #print("LOSS:",loss_ctc)
        s_norm=s_ctc * weight_ctc + s_pred * (1-weight_ctc) #+ 0.5 * s_lm

        #min_=torch.min(s_norm,dim=0,keepdim=False)
        max_=torch.max(s_norm,dim=0,keepdim=False)
        #max_val = max(s_norm)        
        del encoder_output
        ###del trellis
        ###del path
        return final_tokens, final_scores, final_tokens[max_.indices].tolist()
