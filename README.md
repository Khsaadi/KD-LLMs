# Selective Knowledge Distillation of Large Language Models

## 1 Environment
```bash
pip3 install -e transformers/
pip3 install torch==2.0.1
pip3 install deepspeed==0.10.0
pip3 install torchvision==0.15.2
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install sentencepiece
pip3 install protobuf==3.20.3
pip3 install peft
```
or
```bash
bash install.sh
```

Our code is based in [this commit](https://github.com/huggingface/transformers/commit/85fde09c97213bf7e8625f83096bb2a9e183f987) of HuggingFace Transformers.

Our code is based on this [code](https://github.com/microsoft/LMOps/tree/main/minillm)

## 2 Data
### 2.1 Resources
+ The training/evaluation intruction-response data before processing can be downloaded from the following links: [dolly](https://huggingface.co/datasets/MiniLLM/dolly) and [self-inst](https://huggingface.co/datasets/MiniLLM/self-inst)
+ The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext). For reproducibility, we recommend you to use the following preprocessed data.
+ The processed data can be downloaded from the following links: [dolly](https://huggingface.co/datasets/MiniLLM/dolly-processed), [openwebtext](https://huggingface.co/datasets/MiniLLM/openwebtext-processed), 


### 2.2 Data Processing
Get plain-text corpus $\mathcal{D}_\text{PT}$:
```bash
python3 tools/get_openwebtext.py
```
This script will replace the continuous `\n` in each document with a special token "<@x(x!>" and write each document in OpenWebText in a line, which is covenient for parallel processing. In `data/openwebtext/data.txt`, we give an example of the resulting format. You can follow this format to prepare other corpus beyond OpenWebText.

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process OpenWebText Train / Validation Data

bash scripts/opt/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/opt/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process RoBERTa Corpus Train / Validation Data

bash scripts/llama/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/llama/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process RoBERTa Corpus Train / Validation Data
```

## 3 Models
### 3.1 Resources
+ The pre-trained models (MiniLLM and the baselines) can be found in this [collection](https://huggingface.co/collections/MiniLLM/minillm-66f51b3d667b4ee25046dafd).

#### Base Pre-trained Models
To run fine-tuning or standard KD baselines, you need to download the model checkpoints from [Huggingface Model Hub] and put them in `checkpoints/`. For example, for gpt2-large, you can download the model from this [link](https://huggingface.co/gpt2-large/tree/main) and put them in `checkpoints/gpt2-large`.

Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.


## 4 Train
All our experiments are conducted on 1 \* A100 each for about 13 hours. You can set the hyper-parameters in each .sh script.

### Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher model
```bash
bash bash scripts/gpt2/sft/sft_xlarge.sh  /PATH/TO/KD-LLMs
```
#### Fine-tune the student model
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH/TO/KD-LLMs
```
#### Run distillation
```bash
bash scripts/gpt2/minillm/train_base_xl.sh /PATH/TO/KD-LLMs
```
## 5 Run Evaluation
Before running the evaluation, create a directory called gpt2-base and put your final distilled model inside it.
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH/TO/KD-LLMs
```


