#!/bin/bash
set -e

# DON'T NEED $1!
# usage="Usage: bash ${0} batch_file"
# if [ $# -eq 0 ] || [ $1 = '-h' ] ; then
# 	echo $usage
# 	exit 1
# fi

[ -a pkl_model ] || mkdir pkl_model
[ -a pkl_bkp ] || mkdir pkl_bkp

# st=` cat $1 | cut -d# '/' -f 3 | sort | uniq `
# echo $st
# [ $st != time ] && [ $st != user ] && echo $usage && exit 1

# mode type
for st in time user ; do
	for mt in s t st base ; do
		bash run_batch.sh $st $mt run_batch/run_batch_${st}
	done
done

[ ` ls done* | wc -l ` -eq 36 ] && echo -e '\n---------------------------------\n    All jobs have been done!\n---------------------------------\n'
