# UEM: User Experience Model for Sightseeing-Spot Recommendation

This repository contains dataset based on flickr and code that implements the user experience model(UEM) and pseudo-rating mechanism described in [1].

## Methodology

#### Data splitting

We provide two approaches to split datasets, time-split and user-split. The time-split method regards tourist records of each user as a sequence and sets a part of the sequence as the training data and the remaining part as the testing data to simulate a scenario wherein the past records of all users are known and used to recommend new locations to them.  In contrast, the user-split method selects a number of users as the training group and the others as the test group to simulate the cold-start situation wherein there is no knowledge concerning new tourists.

#### User Experience Model

UEM is a probabilistic generation model for analyzing tourist behaviors in sightseeing spots, which involves four factors: Who (does it), What (the tourist does), Where (the tourist does it), and When (the tourist does it). With UEM, we could reveal "what we can do/enjoy there" and recommend sightseeing resources based on user behavior. We provide four variations of UEM.

* **B-UEM**: Basic UEM.
* **S-UEM**: Spatial UEM
* **T-UEM**: Temporal UEM
* **ST-UEM**: Spatio-Temporal UEM

#### Pseudo-rating mechanism

The pseudo-rating mechanism is proposed to handle the cold-start scenario, wherein no historical data exists on the tourists new to the city. Several keywords concerning "Where" and "What" are provided to new tourists to be rated as start-up information, and recommendations are made based on the UEM. We provide four recommendation methods, 

- **base**: only UEM without pseudo-rating
- **al**:  UEM and pseudo-rating with given locations
- **wtol**: UEM and pseudo-rating with given words
- **wl**: UEM and pseudo-rating with given both words and locations

The base method will not be conducted if the data-splitting approach is user-split.

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
Evaluation of spot recommendations can be conducted automatically once the learning process ends. The precision, recall, F-measure and Gini index of four methods (base, al, wtol and wl) at the top k (k=1, 5, 10, 15, 20) recommendations will be printed.

If you want to modify the evaluation process without changing the learning process, we recommend you to make use of the pkl file created by learning process. The pkl file contains the posteriors, data file name and test ids, which are the input of the evaluation process.

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

[1]Kun Yi, Ryu Yamagishi, Taishan Li, Zhengyang Bai, and Qiang Ma. Recommending POIs for Tourists by UserBehavior Modeling and Pseudo-Rating.[arXiv:2110.06523](https://arxiv.org/abs/2110.06523), 2021.
