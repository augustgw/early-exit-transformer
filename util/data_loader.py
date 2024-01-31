import re
import torch
import torchaudio.transforms as T
import torch.nn.functional as F


def spec_transform(waveform, args):
    spec_t = T.Spectrogram(n_fft=args.n_fft * 2,
                           hop_length=args.hop_length,
                           win_length=args.win_length)
    return spec_t(waveform)


def melspec_transform(waveform, args):
    melspec_t = T.MelScale(sample_rate=args.sample_rate,
                           n_mels=args.n_mels,
                           n_stft=args.n_fft+1)
    return melspec_t(waveform)


def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)


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
        char_map = {}
        index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            char_map[ch] = int(index)
            index_map[int(index)] = ch
        index_map[28] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28  # char_map['']
            else:
                ch = char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(index_map[i.detach().item()])
        return ''.join(string)  # .replace('', ' ')


text_transform = TextTransform()


class CollateFn(object):

    def __init__(self, args):
        self.args = args

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):

        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        tensors, targets = [], []
        t_len = []
        t_source = []
        k = 0
        # Gather in lists, and encode labels as indices
        for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
            label = re.sub(r"<unk>|\[ unclear \]", "", label)
            label = re.sub(r"[#^$?:;.!\[\]]+", "", label)
            if len(label) < self.args.max_utterance_length:
                spec = spec_transform(waveform, self.args)  # .to(device)
                spec = melspec_transform(spec, self.args).to(self.args.device)
                t_source += [spec.size(2)]
                tensors += spec
                del spec
                if self.args.bpe == True:
                    # tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label.lower()) + [sp.eos_id()])
                    tg = torch.LongTensor(
                        [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
                else:
                    tg = torch.LongTensor(
                        text_transform.text_to_int("^"+label.lower()+"$"))
                targets += [tg.unsqueeze(0)]
                t_len += [len(tg)]
                k = k+1
                del waveform
                del label
            else:
                print('REMOVED:', ut_id, ' LAB:', label)

        if tensors:
            tensors = pad_sequence(tensors, 0)
            targets = pad_sequence(targets, PAD_token)
            return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_len), torch.tensor(t_source)
        else:
            return None


class CollatePaddingFn(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        # Gather in lists, and encode labels as indices
        batch = sorted(batch, key=lambda x: x[0].size(1), reverse=True)

        n_split = self.args.n_batch_split
        s_sum = sum(x[0].size(1) for x in batch) / n_split
        p_sum = 0
        chunked_batch = list()
        init = 0
        end = 0
        p_split = 0

        for w, *_ in batch:
            p_sum += w.size(1)

            if p_sum >= s_sum:
                chunked_batch.append(batch[init:end+1])
                p_sum = 0
                p_split += 1
                init = end+1

            end += 1

        if p_split != n_split:
            chunked_batch.append(batch[init:end])

        out_batch = []
        for c_batch in chunked_batch:
            tensors, targets, t_len, t_source, o_batch = [], [], [], [], []
            k = 0

            for waveform, smp_freq, label, spk_id, ut_id, *_ in c_batch:
                label = re.sub(r"<unk>|\[ unclear \]", "", label)
                label = re.sub(r"[#^$?:;.!\[\]]+", "", label)

                if len(label) < self.args.max_utterance_length:
                    spec = spec_transform(waveform, self.args)  # .to(device)
                    spec = melspec_transform(
                        spec, self.args).to(self.args.device)
                    t_source += [spec.size(2)]
                    tensors += spec
                    del spec

                    if self.args.bpe == True:
                        tg = torch.LongTensor(
                            [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
                    else:
                        tg = torch.LongTensor(
                            text_transform.text_to_int("^"+label.lower()+"$"))
                    targets += [tg.unsqueeze(0)]
                    t_len += [len(tg)]

                    k = k + 1
                    del waveform
                    del label

                else:
                    print('REMOVED:', ut_id, ' LAB:', label)

            if tensors:
                tensors = pad_sequence(tensors, 0)
                targets = pad_sequence(targets, PAD_token)
                o_batch = [tensors.squeeze(1), targets.squeeze(1),
                           torch.tensor(t_len), torch.tensor(t_source)]

            out_batch.append(o_batch)

        return out_batch
        # return c_tensors, c_targets, c_t_len, c_t_source


class CollateInferFn(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        tensors, targets, t_source = [], [], []

        # Gather in lists, and encode labels as indices
        for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
            label = re.sub(r"[#^$,?:;.!]+|<unk>", "", label)

            if "ignore_time_segment_in_scoring" in label:
                continue
            spec = spec_transform(waveform)  # .to(self.args.device)
            spec = melspec_transform(spec).to(self.args.device)
            t_source += [spec.size(2)]

            npads = 1000
            if spec.size(2) > 1000:
                npads = 500

            tensors += spec
            del spec
            
            if self.args.bpe == True:
                tg = torch.LongTensor(
                    [self.args.sp.bos_id()] + self.args.sp.encode_as_ids(label) + [self.args.sp.eos_id()])
            else:
                tg = torch.LongTensor(
                    text_transform.text_to_int("^"+label.lower()+"$"))

            targets += [tg.unsqueeze(0)]
            del waveform
            del label

        if tensors:
            tensors = pad_sequence(tensors, 0)
            targets = pad_sequence(targets, PAD_token)
            len_out = torch.full((len(t_source),), tensors.size(2))
            
            if self.args.decoder_mode == "aed":
                return tensors.squeeze(1), targets.squeeze(1), len_out
            elif self.args.decoder_mode == "ctc":
                return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_source)

        else:
            return None
