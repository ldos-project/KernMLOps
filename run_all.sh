rm -f *.rdb
cp overrides-thp-always.yaml overrides.yaml
cp config/redis-save.conf config/redis.conf
COLLECTION_PREFIX=update_75_read_5_thp_always_save_on_1 make collect exp0.log 2>&1

rm -f *.rdb
cp overrides-thp-always.yaml overrides.yaml
cp config/redis-no-save.conf config/redis.conf
COLLECTION_PREFIX=update_75_read_5_thp_always_save_off_1 make collect exp1.log 2>&1

rm -f *.rdb
cp overrides-thp-never.yaml overrides.yaml
cp config/redis-save.conf config/redis.conf
COLLECTION_PREFIX=update_75_read_5_thp_never_save_on_1 make collect exp2.log 2>&1

rm -f *.rdb
cp overrides-thp-never.yaml overrides.yaml
cp config/redis-no-save.conf config/redis.conf
COLLECTION_PREFIX=update_75_read_5_thp_never_save_off_1 make collect exp3.log 2>&1
