c=1
for f in w-*; do
    mv -v "$f" "w-$(printf '%0*d' 5 $c).jpeg"
    c=$(($c+1))
done