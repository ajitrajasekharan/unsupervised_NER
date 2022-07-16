rm -f /tmp/vocabs
for i in `ls -1v vocabs`
do
    echo $i
    cat vocabs/$i | tr 'A-Z' 'a-z' | grep -v "#" | awk '{if (length($1) > 2) print $0}' | grep -e "[a-z]"  >> /tmp/vocabs
done
sort -u /tmp/vocabs > merged_vocab.txt
