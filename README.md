# DQ-LoRe
Open Source Code for 'DQ-LoRe: Dual Queries with Low Rank Approximation Re-ranking for In-Context Learning' Submitted to ICLR 2024
![image](https://github.com/UGUESS-lzx/DQ-LoRe/assets/63826387/2e033083-1098-440a-b2a4-02e05e20a442)

The code framework has been modified based on [CEIL](https://github.com/HKUNLP/icl-ceil) , and we are very grateful for their previous work.


## Setup
All required packages can be found in ``requirements.txt``. 
You can install them in a new environment with 
```shell
conda create -n icl python=3.7
conda activate icl

git clone https://github.com/UGUESS-lzx/DQ-LoRe.git

# The following line to be replaced depending on your cuda version.
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

cd DQ-LoRe
pip install -r requirements.txt
# if you don't want to use API from openai, just comment out the `openai` package in `requirements.txt`.
```


## Easy start
```shell
nohup sh script/run_DQ-LoRe.sh > result.out 2>&1 &
```
