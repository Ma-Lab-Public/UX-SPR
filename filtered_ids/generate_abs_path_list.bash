#!/bin/bash

out_put=file_list.txt

[ -a ${out_put} ] && rm ${out_put}

for each in ` ls *txt ` ; do
	echo `pwd`/${each} >> ${out_put}
done

echo Successfullt write abslute path to ${out_put}!
