cat $2 | (while read p; do $1 --file $3 $p; done)
