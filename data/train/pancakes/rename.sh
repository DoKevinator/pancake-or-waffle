c=1
for f in p-*; do
    mv -v "$f" "p-$(printf '%0*d' 5 $c).jpeg"
    c=$(($c+1))
done
