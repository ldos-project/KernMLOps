import sys

file = sys.argv[1]
out = ""
#out2 = ""
with open(file) as f:
    for line in f.readlines():
#        out2 += line
        if ".file" not in line and ".loc" not in line:
            out += line 

with open(file, 'w') as f:
    f.write(out)

#with open(file + "xyz", 'w') as f:
#    f.write(out2)
