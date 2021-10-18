#!/bash/bin

# configuration for bashrc and vimrc
cp ./.bashrc ~/
cp ./.vimrc ~/

# configuration for aws configuration
cp -r ./.aws ~/

# install miniconda
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
#rm Miniconda3-latest-Linux-x86_64.sh

~/miniconda3/bin/conda init bash
#export PATH="/home/ubuntu/miniconda3/bin:$PATH"
export s3='s3://kyoto-shi-photos/group_share/result0916/'

# install aws-cli for conda base
# conda install -n base nodejs=14.7 -y -c conda-forge
# pip install awscli -U
# pip install jupyter -U

