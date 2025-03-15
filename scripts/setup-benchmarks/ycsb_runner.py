#!/usr/bin/python3
#
# Copyright (c) 2012 - 2015 YCSB contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You
# may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License. See accompanying
# LICENSE file.
#

import argparse
import errno
import fnmatch
import io
import os
import shlex
import subprocess
import sys

BASE_URL = "https://github.com/brianfrankcooper/YCSB/tree/master/"
COMMANDS = {
    "shell" : {
        "command"     : "",
        "description" : "Interactive mode",
        "main"        : "site.ycsb.CommandLine",
    },
    "load" : {
        "command"     : "-load",
        "description" : "Execute the load phase",
        "main"        : "site.ycsb.Client",
    },
    "run" : {
        "command"     : "-t",
        "description" : "Execute the transaction phase",
        "main"        : "site.ycsb.Client",
    },
}


DATABASES = {
    "accumulo"     : "site.ycsb.db.accumulo.AccumuloClient",
    "accumulo1.6"     : "site.ycsb.db.accumulo.AccumuloClient",
    "accumulo1.7"     : "site.ycsb.db.accumulo.AccumuloClient",
    "accumulo1.8"     : "site.ycsb.db.accumulo.AccumuloClient",
    "aerospike"    : "site.ycsb.db.AerospikeClient",
    "arangodb"     : "site.ycsb.db.arangodb.ArangoDBClient",
    "arangodb3"     : "site.ycsb.db.arangodb.ArangoDBClient",
    "asynchbase"   : "site.ycsb.db.AsyncHBaseClient",
    "azurecosmos" : "site.ycsb.db.AzureCosmosClient",
    "azuretablestorage" : "site.ycsb.db.azuretablestorage.AzureClient",
    "basic"        : "site.ycsb.BasicDB",
    "basicts"      : "site.ycsb.BasicTSDB",
    "cassandra-cql": "site.ycsb.db.CassandraCQLClient",
    "cassandra2-cql": "site.ycsb.db.CassandraCQLClient",
    "cloudspanner" : "site.ycsb.db.cloudspanner.CloudSpannerClient",
    "couchbase"    : "site.ycsb.db.CouchbaseClient",
    "couchbase2"   : "site.ycsb.db.couchbase2.Couchbase2Client",
    "crail"        : "site.ycsb.db.crail.CrailClient",
    "dynamodb"     : "site.ycsb.db.DynamoDBClient",
    "elasticsearch": "site.ycsb.db.ElasticsearchClient",
    "elasticsearch5": "site.ycsb.db.elasticsearch5.ElasticsearchClient",
    "elasticsearch5-rest": "site.ycsb.db.elasticsearch5.ElasticsearchRestClient",
    "foundationdb" : "site.ycsb.db.foundationdb.FoundationDBClient",
    "geode"        : "site.ycsb.db.GeodeClient",
    "googlebigtable"  : "site.ycsb.db.GoogleBigtableClient",
    "googledatastore" : "site.ycsb.db.GoogleDatastoreClient",
    "griddb"       : "site.ycsb.db.griddb.GridDBClient",
    "hbase098"     : "site.ycsb.db.HBaseClient",
    "hbase10"      : "site.ycsb.db.HBaseClient10",
    "hbase12"      : "site.ycsb.db.hbase12.HBaseClient12",
    "hbase14"      : "site.ycsb.db.hbase14.HBaseClient14",
    "hbase20"      : "site.ycsb.db.hbase20.HBaseClient20",
    "hypertable"   : "site.ycsb.db.HypertableClient",
    "ignite"       : "site.ycsb.db.ignite.IgniteClient",
    "ignite-sql"   : "site.ycsb.db.ignite.IgniteSqlClient",
    "infinispan-cs": "site.ycsb.db.InfinispanRemoteClient",
    "infinispan"   : "site.ycsb.db.InfinispanClient",
    "jdbc"         : "site.ycsb.db.JdbcDBClient",
    "kudu"         : "site.ycsb.db.KuduYCSBClient",
    "memcached"    : "site.ycsb.db.MemcachedClient",
    "maprdb"       : "site.ycsb.db.mapr.MapRDBClient",
    "maprjsondb"   : "site.ycsb.db.mapr.MapRJSONDBClient",
    "mongodb"      : "site.ycsb.db.MongoDbClient",
    "mongodb-async": "site.ycsb.db.AsyncMongoDbClient",
    "nosqldb"      : "site.ycsb.db.NoSqlDbClient",
    "orientdb"     : "site.ycsb.db.OrientDBClient",
    "postgrenosql" : "site.ycsb.postgrenosql.PostgreNoSQLDBClient",
    "rados"        : "site.ycsb.db.RadosClient",
    "redis"        : "site.ycsb.db.RedisClient",
    "rest"         : "site.ycsb.webservice.rest.RestClient",
    "riak"         : "site.ycsb.db.riak.RiakKVClient",
    "rocksdb"      : "site.ycsb.db.rocksdb.RocksDBClient",
    "s3"           : "site.ycsb.db.S3Client",
    "solr"         : "site.ycsb.db.solr.SolrClient",
    "solr6"        : "site.ycsb.db.solr6.SolrClient",
    "tarantool"    : "site.ycsb.db.TarantoolClient",
    "tablestore"   : "site.ycsb.db.tablestore.TableStoreClient"
}

OPTIONS = {
    "-P file"        : "Specify workload file",
    "-p key=value"   : "Override workload property",
    "-s"             : "Print status to stderr",
    "-target n"      : "Target ops/sec (default: unthrottled)",
    "-threads n"     : "Number of client threads (default: 1)",
    "-cp path"       : "Additional Java classpath entries",
    "-jvm-args args" : "Additional arguments to the JVM",
}

def usage():
    output = io.StringIO()
    print("%s command database [options]" % sys.argv[0], file=output)

    print("\nCommands:", file=output)
    for command in sorted(COMMANDS.keys()):
        print("    %s %s" % (command.ljust(14),
                            COMMANDS[command]["description"]), file=output)

    print("\nDatabases:", file=output)
    for db in sorted(DATABASES.keys()):
        print("    %s %s" % (db.ljust(14), BASE_URL +
                            db.split("-")[0]), file=output)

    print("\nOptions:", file=output)
    for option in sorted(OPTIONS.keys()):
        print("    %s %s" % (option.ljust(14), OPTIONS[option]), file=output)

    print("""\nWorkload Files:
    There are various predefined workloads under workloads/ directory.
    See https://github.com/brianfrankcooper/YCSB/wiki/Core-Properties
    for the list of workload properties.""", file=output)

    return output.getvalue()

# Python 2.6 doesn't have check_output. Add the method as it is in Python 2.7
# Based on https://github.com/python/cpython/blob/2.7/Lib/subprocess.py#L545
def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = subprocess.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output


def debug(message):
    print("[DEBUG] ", message, file=sys.stderr)

def warn(message):
    print("[WARN] ", message, file=sys.stderr)

def error(message):
    print("[ERROR] ", message, file=sys.stderr)

def find_jars(dir, glob='*.jar'):
    jars = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for filename in fnmatch.filter(filenames, glob):
            jars.append(os.path.join(dirpath, filename))
    return jars

def get_ycsb_home():
    dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    while "LICENSE.txt" not in os.listdir(dir):
        dir = os.path.join(dir, os.path.pardir)
    return os.path.abspath(dir)

def is_distribution():
    # If there's a top level pom, we're a source checkout. otherwise a dist artifact
    return "pom.xml" not in os.listdir(get_ycsb_home())

# Run the maven dependency plugin to get the local jar paths.
# presumes maven can run, so should only be run on source checkouts
# will invoke the 'package' goal for the given binding in order to resolve intra-project deps
# presumes maven properly handles system-specific path separators
# Given module is full module name eg. 'core' or 'couchbase-binding'
def get_classpath_from_maven(module):
    try:
        debug("Running 'mvn -pl site.ycsb:" + module + " -am package -DskipTests "
            "dependency:build-classpath -DincludeScope=compile -Dmdep.outputFilterFile=true'")
        os.chdir(get_ycsb_home())
        cwd_output = subprocess.check_output(["pwd"])
        debug(cwd_output)
        mvn_output = subprocess.check_output(["mvn", "-pl", "site.ycsb:" + module,
                                "-am", "package", "-DskipTests",
                                "dependency:build-classpath",
                                "-DincludeScope=compile",
                                "-Dmdep.outputFilterFile=true"], universal_newlines=True)
        line = [x for x in mvn_output.splitlines() if x.startswith("classpath=")][-1:]
        return line[0][len("classpath="):]
    except subprocess.CalledProcessError as err:
        error("Attempting to generate a classpath from Maven failed "
            f"with return code '{err.returncode}'. The output from "
            "Maven follows, try running "
            "'mvn -DskipTests package dependency:build=classpath' on your "
            "own and correct errors." + os.linesep + os.linesep + "mvn output:" + os.linesep
            + err.output)
        sys.exit(err.returncode)


def main():
    p = argparse.ArgumentParser(
            usage=usage(),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-cp', dest='classpath', help="""Additional classpath
                entries, e.g.  '-cp /tmp/hbase-1.0.1.1/conf'. Will be
                prepended to the YCSB classpath.""")
    p.add_argument("-jvm-args", default=[], type=shlex.split,
                help="""Additional arguments to pass to 'java', e.g.
                '-Xmx4g'""")
    p.add_argument("command", choices=sorted(COMMANDS),
                help="""Command to run.""")
    p.add_argument("database", choices=sorted(DATABASES),
                help="""Database to test.""")
    args, remaining = p.parse_known_args()
    ycsb_home = get_ycsb_home()

    # Use JAVA_HOME to find java binary if set, otherwise just use PATH.
    java = "java"
    java_home = os.getenv("JAVA_HOME")
    if java_home:
        java = os.path.join(java_home, "bin", "java")
    db_classname = DATABASES[args.database]
    command = COMMANDS[args.command]["command"]
    main_classname = COMMANDS[args.command]["main"]

    # Classpath set up
    binding = args.database.split("-")[0]

    if binding == "accumulo":
        warn("The 'accumulo' client has been deprecated in favor of version "
            "specific bindings. This name still maps to the binding for "
            "Accumulo 1.6, which is named 'accumulo-1.6'. This alias will "
            "be removed in a future YCSB release.")
        binding = "accumulo1.6"

    if binding == "accumulo1.6":
        warn("The 'accumulo1.6' client has been deprecated because Accumulo 1.6 "
            "is EOM. If you are using Accumulo 1.7+ try using the 'accumulo1.7' "
            "client instead.")

    if binding == "cassandra2":
        warn("The 'cassandra2-cql' client has been deprecated. It has been "
            "renamed to simply 'cassandra-cql'. This alias will be removed"
            " in the next YCSB release.")
        binding = "cassandra"

    if binding == "couchbase":
        warn("The 'couchbase' client has been deprecated. If you are using "
            "Couchbase 4.0+ try using the 'couchbase2' client instead.")

    if binding == "hbase098":
        warn("The 'hbase098' client has been deprecated because HBase 0.98 "
            "is EOM. If you are using HBase 1.2+ try using the 'hbase12' "
            "client instead.")

    if binding == "hbase10":
        warn("The 'hbase10' client has been deprecated because HBase 1.0 "
            "is EOM. If you are using HBase 1.2+ try using the 'hbase12' "
            "client instead.")

    if binding == "arangodb3":
        warn("The 'arangodb3' client has been deprecated. The binding 'arangodb' "
            "now covers every ArangoDB version. This alias will be removed "
            "in the next YCSB release.")
        binding = "arangodb"

    if is_distribution():
        db_dir = os.path.join(ycsb_home, binding + "-binding")
        # include top-level conf for when we're a binding-specific artifact.
        # If we add top-level conf to the general artifact, starting here
        # will allow binding-specific conf to override (because it's prepended)
        cp = [os.path.join(ycsb_home, "conf")]
        cp.extend(find_jars(os.path.join(ycsb_home, "lib")))
        cp.extend(find_jars(os.path.join(db_dir, "lib")))
    else:
        warn("Running against a source checkout. In order to get our runtime "
            "dependencies we'll have to invoke Maven. Depending on the state "
            "of your system, this may take ~30-45 seconds")
        db_location = "core" if (binding == "basic" or binding == "basicts") else binding
        project = "core" if (binding == "basic" or binding == "basicts") else binding + "-binding"
        db_dir = os.path.join(ycsb_home, db_location)
        # goes first so we can rely on side-effect of package
        maven_says = get_classpath_from_maven(project)
        # TODO when we have a version property, skip the glob
        cp = find_jars(os.path.join(db_dir, "target"),
                    project + "*.jar")
        # already in jar:jar:jar form
        cp.append(maven_says)
    cp.insert(0, os.path.join(db_dir, "conf"))
    classpath = os.pathsep.join(cp)
    if args.classpath:
        classpath = os.pathsep.join([args.classpath, classpath])

    ycsb_command = ([java] + args.jvm_args +
                    ["-cp", classpath,
                    main_classname, "-db", db_classname] + remaining)
    if command:
        ycsb_command.append(command)
    print(" ".join(ycsb_command), file=sys.stderr)
    try:
        return subprocess.call(ycsb_command)
    except OSError as e:
        if e.errno == errno.ENOENT:
            error('Command failed. Is java installed and on your PATH?')
            return 1
        else:
            raise

if __name__ == '__main__':
    sys.exit(main())
