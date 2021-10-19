## Build  enviroment
```bash
pip3 install -f requirement.txt -U
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
## Caculate perplexity
#### 1. Copy pkl model from ./pkl_bkp
```bash
mv ./pkl_bkp/* ./pkl_model/
```
#### 2. Run perplexisty caculation
```bash
python(3) ./src/perplexsity/calc_perplexity_with_pyro_time_split.py
```
or
```bash
python(3) ./src/perplexsity/calc_perplexity_with_pyro_user_split.py
```
