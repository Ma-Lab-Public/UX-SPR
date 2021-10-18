#!/bin/bash
set -e
usage="Usage: bash ${0} time|user s|t|st|base batch_file"
s3='s3://kyoto-shi-photos/group_share/result0916'
s3_done="${s3}/done/"
s3_model="${s3}/pkl_model/"
s3_doing="${s3}/doing/"

if [ $# -eq 0 ] || [ $1 = '-h' ] ; then
	echo $usage && exit 1
fi

[ $1 != 'time' ] && [ $1 != 'user' ] && echo "Error: please type in time|user for mode!" && exit 1  
[ $2 != 's' ] && [ $2 != 't' ] && [ $2 != 'st' ] && [ $2 != 'base' ] && echo 'Error: please type in s|t|st|base for model!' && exit 1
[ $2 = 's' ] || [ $2 = 't' ] && tag=_${2}_ || tag=${2}
if ! [ -a $3 ] ; then
	echo $usage && exit 1
fi

echo Run learning mode: ${1} >&2
echo "   " model: ${2} >&2

total_n=$( cat $3 | grep -c '\-\-' )
echo  $total_n samples need to be done >&2
n=0
log=run_learning.${2}.${1}_split.${3##*/}.log
echo -E 'Split mode:' $1 '\nModel type:' $2 > $log 

# [ -e ${log} ] && rm ${log}

for line in $( cat $3 ) ; do

        n=$((n+1))
        each=(`echo $line | sed 's/\-\-/ /g'`)
		f=${each[0]}
		r=${each[1]}
        s=${each[2]}

		doing_f=doing_${f##.*/}_${s}_${r}_split_by_${1}_$tag
		# touch ${doing_f} && aws s3 cp ${doing_f} ${s3_doing}
		aws s3 ls ${s3_doing} > temp_doing
		if [ `grep ${doing_f} temp_doing | wc -l` -eq 1 ] ; then
			echo '<--skip-->' $@ $line 
			continue
		else
			touch ${doing_f} && aws s3 cp ${doing_f} ${s3_doing}
		fi

		echo ------------------------ ${n}/${total_n} ---------------------------- >> ${log}

        # each=(`echo $line | sed 's/\-\-/ /g'`)
		# f=${each[0]}
		# r=${each[1]}
        # s=${each[2]}
        echo $( date )' START: ' $( echo $i | awk -F'/' '{print $NF}' ) >> ${log}
        # already change id_data to user-split 
		python ./src/learning/${1}/new_${2}_for_sightseeing.split_by_${1}.py -f ${f} -s ${s} -t ${r} split_by_${1} $tag 
        echo $( date )' FINISH: ' $( echo $i | awk -F'/' '{print $NF}' ) >> ${log}

		done_f=done_${f##.*/}_${s}_${r}_split_by_${1}_$tag
		touch ${done_f} && aws s3 cp ${done_f} ${s3_done}
		aws s3 sync ./pkl_model/ ${s3_model}
		[ `ls ./pkl_model/ | wc -l` -eq 0 ] || mv ./pkl_model/* ./pkl_bkp/
done

#aws s3 sync ./pkl_model/ ${s3_model}
#[ `ls ./pkl_model/ | wc -l` -eq 0 ] || mv ./pkl_model/* ./pkl_bkp/
