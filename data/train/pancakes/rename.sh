c=1
for f in *.jpg; do
    mv -v "$f" "p-$(printf '%0*d' 5 $c).jpg"
    c=$(($c+1))
done
