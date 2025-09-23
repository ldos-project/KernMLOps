while read p; do
    $1 $p -file $3
done <$2
