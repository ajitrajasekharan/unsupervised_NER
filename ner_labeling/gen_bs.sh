input=${1?"Specify input list"}
python ./construct_bs.py -list $input
sort bootstrap_entities.txt  > tmp_bs.txt
mv tmp_bs.txt  bootstrap_entities.txt
#cat bootstrap_entities.txt | awk '{print $1}' | sed 's/\//\n/g' | sort -u > unique_labels.txt
cat bootstrap_entities.txt | awk '{print $1}' | sed 's/\//\n/g' | sort  | uniq -c | sort -k1,1 -n -r >  unique_labels.txt
wc -l bootstrap_entities.txt
wc -l unique_labels.txt
