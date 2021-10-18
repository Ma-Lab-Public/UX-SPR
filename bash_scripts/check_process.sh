#!/bin/bash

cat `ls -t run_learning*log`
echo Finished: ` ls done_* | wc -l`/36
