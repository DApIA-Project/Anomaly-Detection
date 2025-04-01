import os


# Cleaning !

# remove pycache
"find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete"
# remove py
"find . -type f -iname \*.py -delete"
# remove empty folder
"find . -type d -empty -delete"

os.system('ssh ovh2 "cd dapia/ && find . -type f -name \'*.py[co]\' -delete -o -type d -name __pycache__ -delete && find . -type f -iname \*.py -delete && find . -type d -empty -delete"')



f = []
n = 0
for path, subdirs, files in os.walk('./'):
    for name in files:
        p = os.path.join(path, name)
        n += 1
        if p.endswith('.py'):
            f.append(p)


# remove ../tmp folder
os.system('rm -r ../tmp/*')


# copy these files to the folder "../tmp" by respecting directory three
for file in f:
    folder = file.split('/')
    folder =  "/".join(folder[1:-1])
    file = file.split('/')[-1]
    if (folder == ''):
        os.system('cp ' + file + ' ../tmp/'+file)
    else:
        os.system("mkdir -p ../tmp/" + folder)
        os.system('cp ' + folder+'/'+file + ' ../tmp/'+folder+'/'+file)

# os.system('scp -r ../tmp/* ml:WORK/DAPIA/')
os.system('scp -r ../tmp/* ovh2:dapia/')


# # connect shh to the server and execute cd WORK/DAPIA/ && ./main.sh
# os.system('ssh ml "cd WORK/DAPIA/ && ./main.sh"')

