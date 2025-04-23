#!/bin/bash

replace_always() {
    sed -e "/transparent_hugepages/,/$/s/always/$1/" -i $2
}

cp config/redis_always.yaml config/redis_never.yaml &&
    replace_always never config/redis_never.yaml

cp config/redis_always.yaml config/redis_madvise.yaml &&
    replace_always madvise config/redis_madvise.yaml
