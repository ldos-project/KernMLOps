import sys

file = sys.argv[1]
out = ""
with open(file) as f:
    for line in f.readlines():
        if ".file" not in line and ".loc" not in line:
            out += line 

with open(file, 'w') as f:
    f.write(out)
