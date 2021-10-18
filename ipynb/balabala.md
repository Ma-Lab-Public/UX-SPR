## This is private work flow for UEM

## Enviroment configuration

git clone https://github.com/707728642li/work_flow_UEM.git

cd work_flow_UEM/env && bash config_env.sh && . ~/miniconda3/bin/activate

cd .. && conda install python=3.8 -y && pip install -r requirement.txt -U && pip install awscli -U && echo "Enviroment configuration done!"


## Run learning

bash run_all.bash ./run_batch/run_batchXXX

