import os

for d in ["datasets/","models/"]:
    for i in ['drebin','ember','pdf']:
        if not os.path.exists(os.path.join(d,i)):
            os.makedirs(os.path.join(d,i))

os.makedirs("materials")


