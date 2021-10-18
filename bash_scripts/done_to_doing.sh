#!/bin/bash
set -e
usage="Usage: bash ${0} time|user s|t|st|base batch_file"
s3='s3://kyoto-shi-photos/group_share/result0916'
s3_done="${s3}/done/"
s3_model="${s3}/pkl_model/"
s3_doing="${s3}/doing/"

mkdir temp_doing

aws s3 ls $s3_done | sed 's/.* // ; s/done/doing/' | xargs -I {} touch temp_doing/{}
n=`ls temp_doing | wc -l`
aws s3 sync ./temp_doing/ $s3_doing && rm -rf temp_doing
echo $n done files are identified and transfer to doing file successfully!

