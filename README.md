# Early-Exit Architectures for ASR

Training dynamic [Conformer](https://arxiv.org/abs/2005.08100) models for Automatic Speech Recognition (ASR) using early-exiting training techniques. 

**Paper**

Find our original paper on early-exit training, 'Training dynamic models using early exits for automatic speech recognition on resource-constrained devices', on [arXiv](https://arxiv.org/abs/2309.09546).

**Acknowledgements**

Incorporates code from [Transformer PyTorch implementation by Hyunwoong Ko](https://github.com/hyunwoongko/transformer) and [SentencePiece unsupervised tokenizer](https://github.com/google/sentencepiece).

### Usage

**Basic usage**

| Description          | Command        |
| ----------------- | -------------------- |
| Training an Attention Encoder-Decoder-based model | `train.py --decoder_mode aed` |
| Training a CTC-based model | `train.py --decoder_mode ctc --model_type model_name` |
| Inference with an Attention Encoder-Decoder-based model | `inference.py --decoder_mode aed --load_model_path /path/to/model` |
| Inference with a CTC-based model | `inference.py --decoder_mode ctc --model_type model_name --load_model_path /path/to/model` |

**Advanced usage examples**

| Description          | Command        |
| ----------------- | -------------------- |
| Training an AED-based model with 6 exits, one placed every 3 layers, for a total of 18 layers | `train.py --decoder_mode aed --n_enc_exits 6 --n_enc_layers_per_exit 3` |
| Training a CTC-based model for 75 epochs with an initial learning rate of 1e-6. The model is initialized from a pre-trained model checkpoint found at the given path | `train.py --decoder_mode ctc --model_type model_name --n_epoch 75 --init_lr 1e-6 --load_model_path /path/to/model` |
| Inference with an AED-based architecture, based on the average of model checkpoints from epochs 95 through 100 found in the directory at the given path | `inference.py --decoder_mode aed --load_model_dir /path/to/dir --avg_model_start 95 --avg_model_end 100` |

See below for additional configuration options.

### Configuration

*Note:* [SentencePiece](https://github.com/google/sentencepiece) is used to tokenize target labels.

**Training setup and options**

<!--- | `--bpe`           | `True`               | Whether to use BPE-based tokenization with SentencePiece       |
| `--distill`       | `False`               | Whether to use knowledge distillation       | 
| `--lexicon_path`           | `lexicon.txt`               | Path to lexicon file       |
| `--tokens_path`           | `tokens.txt`               | Path to tokens file      | --->

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--decoder_mode`  | --                 | **Required**: Whether to use a connectionist temporal classification-based (`ctc`) or attention encoder-decoder-based (`aed`) decoder       |
| `--model_type` | `early_conformer`               | Choose the model to use: `early_conformer`, `early_conformer_plus` or `early_zipformer` (Only for `ctc` decoder)    |
| `--n_epochs` | `10000`               | Number of training epochs      |
| `--n_threads` | `10`               | Number of threads for intraop parallelism on CPU. See PyTorch torch.set_num_threads method      |
| `--n_workers` | `10`               | Number of GPU workers for loading data      |
| `--shuffle`       | `True`               | Shuffles training data upon loading       |
| `--save_model_dir`       | `/trained_model`               | Directory in which to save model checkpoints      |
| `--load_model_path`       | `None`               | Path to model checkpoint to load for training/inference       |
| `--load_model_dir`       | `None`               | Directory containing models checkpoints for model averaging       |
| `--avg_model_start`       | `None`               | Starting epoch for model averaging       |
| `--avg_model_end`       | `None`               | End epoch for model averaging      |

*Note:* In addition to the specified number of conformers and layers per conformer, the `early_conformer_plus` model automatically includes one extra parallel downsampled layer (a conformer with a single layer) before both the first and last exits. Which adds a total of two extra layers compared to the `early_conformer` model with the same parameters.

**Model parameters**

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--batch_size`           | `64`               | Batch size during training and inference       |
| `--n_batch_split`           | `4`               | In each batch, items are ordered by length and split into this number of sub-batches, in order to minimize padding and maximize GPU performance      |
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
| `--aed_ce_weight`           | `0.7`               | For AED models: weight coefficient for the cross-entropy loss.       |
| `--aed_ctc_weight`           | `0.3`               | For AED models: weight coefficient for the CTC loss.       |



**Audio preprocessing**

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--sample_rate`           | `16000`               | Sample rate used in preprocessing raw audio inputs to the model       |
| `--n_fft`           | `512`               | Size of Fast Fourier Transform used to generate spectrogram of raw audio input during preprocessing       |
| `--win_length`           | `320`               | Window length used to generate spectrogram of raw audio input during preprocessing       |
| `--hop_length`           | `160`               | Length of hop between STFT windows used to generate spectrogram of raw audio input during preprocessing      |
| `--n_mels`           | `80`               | Number of mel filterbanks used to compute STFT of raw audio input during preprocessing       |

**Optimization**

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--init_lr`           | `1e-5`               | Initial learning rate during training       |
| `--adam_eps`           | `1e-9`               | Epsilon parameter used in AdamW optimization algorithm       |
| `--weight_decay`           | `5e-4`               | Weight decay coefficient used in AdamW optimization algorithm       |
| `--warmup`         | `-1`               | Number of learning rate warmup steps. Default behavior (-1): Warmup for the length of the dataloader.       |
| `--clip`           | `1.0`               | Gradient norms higher than this value will be clipped during training. See PyTorch torch.nn.utils.clip_grad_norm_ function       |

**Inference parameters**

| Variable          | Default value        | Description                    |
| ----------------- | -------------------- | ------------------------------ |
| `--beam_size`           | `10`               | Beam size for AED beam search inference       |
| `--pen_alpha`           | `1.0`               | Sentence length penalty for AED beam search inference       |
