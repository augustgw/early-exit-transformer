from conf import *
import torchaudio.transforms as T
import torch.nn.functional as F
import re

'''
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
'''
'''
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
'''
spec_transform = T.Spectrogram(n_fft=n_fft * 2, \
                               hop_length=hop_length, \
                               win_length=win_length)

melspec_transform = T.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_fft+1)

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        # 30
        ^ 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 29
        $ 31
        @ 0
        """
        # ^=<SOS> 1
        # $=<EOS> 31
        # #=<PAD> 30
        # @=<blank> for ctc
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[28] = ' '
    def text_to_int(self,text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28#self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence
    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i.detach().item()])
        return ''.join(string)#.replace('', ' ')                                     


def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)

text_transform=TextTransform()

def collate_fn(batch, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token=trg_pad_idx):   

    tensors, targets = [], []
    t_len=[]
    t_source=[]
    k = 0
    # Gather in lists, and encode labels as indices
    for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
        label=re.sub(r"<unk>|\[ unclear \]","",label)
        label=re.sub(r"[#^$?:;.!\[\]]+","",label)        
        if len(label) < max_utterance_length:
            spec=spec_transform(waveform)#.to(device)
            spec = melspec_transform(spec).to(device)
            t_source += [spec.size(2)]
            tensors += spec
            del spec
            if bpe_flag == True:
                #tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label.lower()) + [sp.eos_id()])
                tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label) + [sp.eos_id()])                
            else:
                tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
            targets += [tg.unsqueeze(0)]
            t_len += [len(tg)]
            k=k+1
            del waveform
            del label
        else:
            print('REMOVED:',ut_id,' LAB:',label)

    if tensors:
        tensors = pad_sequence(tensors,0)
        targets = pad_sequence(targets,PAD_token)
        return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_len), torch.tensor(t_source)
    else:
        return None


def collate_padding_fn(batch, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token=trg_pad_idx):   

    # Gather in lists, and encode labels as indices
    batch=sorted(batch, key=lambda x: x[0].size(1), reverse=True)

    nsplit = n_batch_split
    s_sum =sum(x[0].size(1) for x in batch) / nsplit
    p_sum=0
    chunked_batch = list()
    init=0
    end=0
    p_split=0
    for w,*_ in batch:
        p_sum += w.size(1)
        if p_sum >= s_sum:
            chunked_batch.append(batch[init:end+1])            
            p_sum = 0
            p_split += 1
            init = end+1
        end += 1
    if p_split != nsplit:
        chunked_batch.append(batch[init:end])        

    out_batch = []        
    for c_batch in chunked_batch:
        tensors, targets, t_len, t_source, o_batch = [], [], [], [], []
        k = 0

        for waveform, smp_freq, label, spk_id, ut_id, *_ in c_batch:
            label=re.sub(r"<unk>|\[ unclear \]","",label)
            label=re.sub(r"[#^$?:;.!\[\]]+","",label)

            if len(label) < max_utterance_length:
                spec=spec_transform(waveform)#.to(device)
                spec = melspec_transform(spec).to(device)
                t_source += [spec.size(2)]
                tensors += spec
                del spec
                if bpe_flag == True:
                    #tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label.lower()) + [sp.eos_id()])
                    tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label) + [sp.eos_id()])
                else:
                    tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
                targets += [tg.unsqueeze(0)]
                t_len += [len(tg)]

                k=k+1
                del waveform
                del label
            else:
                print('REMOVED:',ut_id,' LAB:',label)

        if tensors:
            tensors = pad_sequence(tensors,0)
            targets = pad_sequence(targets,PAD_token)
            o_batch = [tensors.squeeze(1), targets.squeeze(1),torch.tensor(t_len),torch.tensor(t_source)]
            #print("TENSOR:",tensors.squeeze(1).size(), torch.tensor(t_source).size())
        out_batch.append(o_batch)

    '''    
    for o_,a_,b_,c_ in out_batch:
        print("LB:",len(o_),len(a_),len(b_), len(c_))
        for oo_, aa_,bb_,cc_ in zip(o_,a_,b_,c_):
            #print("OB:",len(i_), i_.size())
            print("OB:", oo_.size(), aa_.size(), bb_.size(), cc_.size())            
            print(aa_,bb_,cc_)
    '''
    return out_batch
    #return c_tensors, c_targets, c_t_len, c_t_source
