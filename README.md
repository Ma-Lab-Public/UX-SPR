## Basic infomation/ program description / citation
*Waiting for Yikun*
## Build  environment (under python3.8)
```bash
pip3 install -r requirement.txt -U
```
Configure aws account in *~/.aws/credentials*

## Learning
#### 1. Create result folder
1. Create ***result folder*** in S3 
2. Create ***pkl_model***, ***doing*** and ***done*** folder under result folder    
3. Configure result folder in ./bash_scripts/run_batch.sh line 4

#### 2. Run learning
```bash
bash ./bash_scripts/run_all.sh 
```
## Evaluation
*Waiting for Yikun*
## Calculate perplexsity
#### 1. Copy pkl model from ./pkl_bkp
```bash
cp ./pkl_bkp/* ./pkl_model/
```
#### 2. Run perplexsity caculation
```bash
python(3) ./src/perplexsity/calc_perplexity_with_pyro_time_split.py
```
or
```bash
python(3) ./src/perplexsity/calc_perplexity_with_pyro_user_split.py
```
