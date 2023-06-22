#!/bin/sh


for set in ^test-clean ^dev-clean ^test-other ^dev-other ^asr_dev ^asr_test ^dev ^test
do
    echo $set
    cat $1 |egrep $set |grep EXP |cut -f2- -d":">1
    for i in  1 2 3 4 5 6
    do
	echo $i
	cat $1 |egrep $set|grep "BEAM_OUT_ $i"|cut -f2- -d":">2
	./levenshtein -in1 1 -in2 2 |tail -1 
    done
    echo "Total lines:"`wc 1|awk '{print $1}'`
done

