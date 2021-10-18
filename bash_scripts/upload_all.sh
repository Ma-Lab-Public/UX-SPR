#!/bin/bash
if [ $# -eq 0 ] || [ $1 = '-h'] ; then
	echo bash $0 run_batch_XX
	exit 1
fi

tar -zcvf ../${1}.tar.gz ../work_flow_UEM && aws s3 cp ../${1}.tar.gz s3://kyoto-shi-photos/group_share/result0916/history/ && echo Upload successfully!
