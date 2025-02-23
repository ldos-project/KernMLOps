#!/bin/bash
mongosh <<EOF
use ycsb
db.usertable.drop()
exit
EOF
