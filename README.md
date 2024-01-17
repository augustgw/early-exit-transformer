# Early-Exit Architectures for ASR

Training dynamic [Conformer](https://arxiv.org/abs/2005.08100) models for Automatic Speech Recognition (ASR) using early-exiting training techniques. 

***Paper***

Find our original paper on early-exit training, 'Training dynamic models using early exits for automatic speech recognition on resource-constrained devices', on [arXiv](https://arxiv.org/abs/2309.09546).

***Acknowledgements***

Incorporates code from [Transformer PyTorch implementation by Hyunwoong Ko](https://github.com/hyunwoongko/transformer) and [SentencePiece unsupervised tokenizer](https://github.com/google/sentencepiece).

### Usage

***CTC***

`main.py --decoder_mode ctc --option value`

***Attention Encoder-Decoder***

`main.py --decoder_mode aed --option value`

See below for additional configuration options.

### Configuration

*Note:* [SentencePiece](https://github.com/google/sentencepiece) is used to tokenize target labels.

***Training setup and options***

<!--- | `--bpe`           | `True`               | Whether to use BPE-based tokenization with SentencePiece       |
| `--distill`       | `False`               | Whether to use knowledge distillation       | 
| `--lexicon_path`           | `lexicon.txt`               | Path to lexicon file       |
| `--tokens_path`           | `tokens.txt`               | Path to tokens file      | --->

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--decoder_mode`  | --                 | **Required**: Whether to use a connectionist temporal classification-based (`ctc`) or attention encoder-decoder-based (`aed`) decoder       |
| `--num_threads` | `10`               | Sets number of threads for intraop parallelism on CPU. See PyTorch torch.set_num_threads method      |
| `--num_workers` | `10`               | Sets number of GPU workers for loading data      |
| `--shuffle`       | `True`               | Shuffles training data upon loading       |

***Model parameters***

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--batch_size`           | `64`               | Batch size during training and inference       |
| `--n_batch_split`           | `64`               | In each batch, items are ordered by length and split into this number of sub-batches, in order to minimize padding and maximize GPU performance      |
| `--max_len`       | `2000`               | Maximum length in terms of number of characters for model inputs       |
| `--d_model`           | `256`               | Dimensionality of the model       |
| `--n_enc_layers_per_exit`           | `2`               | Number of encoder layers per exit (where number of exits is determined by --n_enc_exits). For example, --n_enc_layers_per_exit=2 and --n_enc_exits=6 results in a encoder with 6 exits and 12 total layers, with an exit occurring every 2 layers       |
| `--n_enc_exits`           | `6`               | Number of exits in the model (where number of layers per exit is determined by --n_enc_layers_per_exit). For example, --n_enc_layers_per_exit=2 and --n_enc_exits=6 results in a encoder with 6 exits and 12 total layers, with an exit occurring every 2 layers       |
| `--n_dec_layers`           | `6`               | Number of decoder layers in each exit in the encoder      |
| `--n_heads`           | `6`               | Number of attention heads in each encoder layer       |
| `--d_feed_forward`           | `2048`               | Dimensionality of the feed-forward network       |
| `--drop_prob`           | `0.1`               | Probability of a given element of the input to be randomly dropped during training       |
| `--depthwise_kernel_size`           | `31`               | Kernel size of the depthwise convolutions in each Conformer block       |
| `--max_utterance_length`           | `360`               | Input items longer than this value in terms of number of labels will be dropped during training       |


***Audio preprocessing***

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--sample_rate`           | `16000`               | Sample rate used in preprocessing raw audio inputs to the model       |
| `--n_fft`           | `512`               | Size of Fast Fourier Transform used to generate spectrogram of raw audio input during preprocessing       |
| `--win_length`           | `320`               | Window length used to generate spectrogram of raw audio input during preprocessing       |
| `--hop_length`           | `160`               | Length of hop between STFT windows used to generate spectrogram of raw audio input during preprocessing      |
| `--n_mels`           | `80`               | Number of mel filterbanks used to compute STFT of raw audio input during preprocessing       |

***Optimization***

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--init_lr`           | `1e-5`               | Initial learning rate during training       |
| `--adam_eps`           | `1e-9`               | Epsilon parameter used in AdamW optimization algorithm       |
| `--weight_decay`           | `5e-4`               | Weight decay coefficient used in AdamW optimization algorithm       |
| `--warmup`         | `8000`               | Number of learning rate warmup steps       |
| `--clip`           | `1.0`               | Gradient norms higher than this value will be clipped during training. See PyTorch torch.nn.utils.clip_grad_norm_ function       |
