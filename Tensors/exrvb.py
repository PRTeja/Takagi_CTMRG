import numpy as np

def dec(num, base, l):
     dig = np.array([0 for i in range(l)])
     quo = num
     div = base**(l-1)
     for i in range(l):             
             dig[i] = quo//div
             quo    = quo%div
             div = div//base
             #dig[0] is the digit of highest place value
     return dig

a = np.zeros(3**4)

downcount =0
upcount   =0 
for i in range(3**4):
	num = dec(i,3,4)
	count = [0 for j in range(3)]
	for j in range(4):
		if num[j] == 0:
			count[0] +=1
		if num[j] == 1:
			count[1] +=1
		if num[j] == 2:
			count[2] +=1
	
	if (count[0] == 2) and (count[1] == 1) and (count[2] ==1):
		downcount += 1
	
	if (count[2] == 2) and (count[1] == 1) and (count[0] ==1):
		upcount += 1
	
wf = np.zeros((2,3,3,3,3))

for i in range(3**4):
	num = dec(i,3,4)
	count = [0 for j in range(3)]
	for j in range(4):
		if num[j] == 0:
			count[0] +=1
		if num[j] == 1:
			count[1] +=1
		if num[j] == 2:
			count[2] +=1
	
	if (count[0] == 2) and (count[1] == 1) and (count[2] ==1):
		index = [0] +list(num)
		wf[tuple(index)] = 1./np.sqrt(downcount)
	
	if (count[2] == 2) and (count[1] == 1) and (count[0] ==1):
		index = [1] +list(num)
		wf[tuple(index)] = 1./np.sqrt(upcount)

f = open("Tensor2.dat",'w+')
for i0 in range(2):
	for i in range(3**4):
		num = dec(i,3,4)
		index = [i0]+list(num)
		if (wf[tuple(index)]) != 0:
			f.write(str(i0)+'\t'+str(num[0]) + '\t' + str(num[1]) + '\t' + str(num[2]) + '\t' + str(num[3]) + '\t' +str(wf[tuple(index)]) + '\n')
		
f.close()
		
