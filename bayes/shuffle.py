import random
import json
import math

def shuffle(filename):
  j1 = json.load(open(filename), strict=False)
  key = list(j1.keys())
  content = list(j1.values())
  length = len(content)
  l = list(range(length))
  random.shuffle(l)

  nnum = 0
  trainnum = math.floor(length/5*4-1)
  f = open('train.json', 'w')
  f.write('{')
  f.close()
  for i in range(trainnum):
    num = l[i]
    f = open('train.json', 'a')
    if nnum == 0:
      f.write('"'+key[num]+'":')
    else:
      f.write(',"'+key[num]+'":')
    f.close()
    json.dump(content[num], open('train.json', 'a'))
    nnum += 1
  f = open('train.json', 'a')
  f.write('}')
  f.close()
  print(nnum)

  nnum = 0
  valnum = math.floor(length/5)
  #clear
  f = open('val.json', 'w')
  f.write('{')
  f.close()
  for i in range(trainnum,trainnum+valnum):
    num = l[i]
    f = open('val.json', 'a')
    if nnum == 0:
      f.write('"'+key[num]+'":')
    else:
      f.write(',"'+key[num]+'":')
    f.close()
    json.dump(content[num], open('val.json', 'a'))
    nnum += 1
  f = open('val.json', 'a')
  f.write('}')
  f.close()
  print(nnum)

  nnum = 0
  devnum = length - trainnum - valnum
  #clear
  f = open('dev.json', 'w')
  f.write('{')
  f.close()
  for i in range(trainnum+valnum,length):
    num = l[i]
    f = open('dev.json', 'a')
    if nnum == 0:
      f.write('"'+key[num]+'":')
    else:
      f.write(',"'+key[num]+'":')
    f.close()
    json.dump(content[num], open('dev.json', 'a'))
    nnum += 1
  f = open('dev.json', 'a')
  f.write('}')
  f.close()
  print(nnum)