import numpy as np
import scipy
from math import *
import time
from matplotlib import pyplot as plt
from decimal import *
from itertools import groupby
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import optimize as opt
from scipy.linalg import block_diag, sqrtm, polar, schur, pinv
from numpy import sqrt as Sqrt
import os.path
import torch
import tensorflow as tf
import sys
"""
Break down the 2-site operator into 2 one-side operators using svd. create 2 new tensors with it an use it to form 2 corners 
and then use the corners to calculate the expectation value(computational cost ~X^2*D^4*ds^2) instead of calculating the whole environment tensor(D^12)

"""

#np.set_printoptions(suppress = True)
getcontext().prec = 16
#np.set_printoptions(threshold=np.float64)
    
def tsum(string, *args):
	#args = tuple([torch.tensor(np.array(a, dtype = np.complex128)) for a in args])
	
	return tf.einsum(string, *args).numpy() 

def EigenDecomposition(C):
	s,u = scipy.linalg.eig(C)
	sort = np.argsort(np.abs(s))[:-len(s)-1:-1]
	
	return s[sort], u[:,sort]
	

def zz(*args,**kwargs):
     kwargs.update(dtype=np.complex128)
     return np.zeros(*args,**kwargs)  

def EigDec(C):
	s,u = np.linalg.eig(C)
	sort = np.argsort(np.abs(s))[:-len(s)-1:-1]
	s = s[sort]
	u = u[:,sort]
	
	return s,u
	

def EA(A,A1,D,ds):
	D2 = D*D
	
	E = tsum('ijklm, iabcd', A,np.conjugate(A1))
	E = tsum('abcdjklm -> ajbkcldm',E)
	E = np.reshape(E, (D2,D2,D2,D2))
	
	return E

def EA1(A,O,D,ds):
	D2 = D*D
	E = zz((D2,D2,D2,D2))
	A1     = tsum('ij,jklmn', O,A)
	E = tsum('ijklm, iabcd', A1, np.conjugate(A) )
	E = tsum('abcdjklm -> ajbkcldm',E)
	E = np.reshape(E, (D2,D2,D2,D2))				
	return E

def EAcustom(A,A1):
	shape1 = A.shape[1:5]
	shape2 = A1.shape[1:5]
	res    = np.array(shape1)*np.array(shape2)
	
	E      = tsum('aijkl, apqrs -> ipjqkrls', A, np.conjugate(A1)).reshape(res)
	
	return E
def B_singlet(A,D,ds, Bond):
	B = tsum('ik, jklmn -> jilmn', Bond, A)
	B = tsum('il, jklmn -> jkimn', Bond, B)
	B = tsum('im, jklmn -> jklin', Bond, B)
	B = tsum('in, jklmn -> jklmi', Bond, B)
	
	return B

def doubleE(E, EB, D2):
	E2 = tsum('ijkl,mnoj ->imnokl', E,EB)
	return E2


def doubleH2(A,B,D,ds,H):
	# H(1234) = H( i',i; j',j) 
	
	D2 = D*D
	u,s,vt = np.linalg.svd(H)
	u  = np.reshape(u, (ds,ds,ds*ds))  # u(i',i, k)
	vt = np.reshape(vt,(ds*ds,ds,ds))
	vt = tsum('ij,jkl ->ikl', np.diag(s), vt)
	
	sqr = sqrtm(H)
	u = np.reshape(sqr,(ds,ds,ds*ds))
	vt = np.reshape(sqr,(ds*ds,ds,ds))
		
	Au = tsum('ijk,jlmno ->kilmno', u, A)
	Bu = tsum('ijk,klmno ->ijlmno', vt,B)
	
	AuA = tsum('oiabcd, ijklm -> oajbkcldm',Au, np.conjugate(A))
	BuB = tsum('oiabcd, ijklm -> oajbkcldm',Bu, np.conjugate(B))
	AuA = np.reshape(AuA, (ds*ds, D2,D2,D2,D2) )
	BuB = np.reshape(BuB, (ds*ds, D2,D2,D2,D2) )	

	return AuA, BuB		


	
def doubleH(A,B,D,ds,H):
				
	D2 = D*D
	
	A2 = tsum('abcde,ijklc -> aibjklde', A, B)					
	A2 = A2.reshape(ds*ds,D,D,D,D,D,D)
	print(H.shape)
	A2H = tsum('ij,jklmnop->iklmnop', H,A2)
	E2  = tsum('ijklmno,ipqrstu ->jpkqlrmsntou', A2H, np.conjugate(A2))
	E2  = np.reshape(E2, (D2,D2,D2,D2,D2,D2))
	
	return E2		
	
def fixE(E2, Bt2, Bt2in):
	print(Bt2in.shape, E2.shape, Bt2.shape)
	E20 = tsum('ai, ijklmn -> ajklmn', Bt2in, E2)
	E20 = tsum('aj, ijklmn -> iaklmn', Bt2, E20)
	E20 = tsum('ak, ijklmn -> ijalmn', Bt2, E20)	
	E20 = tsum('al, ijklmn -> ijkamn', Bt2, E20)	
	E20 = tsum('am, ijklmn -> ijklan', Bt2in, E20)
	E20 = tsum('an, ijklmn -> ijklma', Bt2in, E20)
	
	return E20
	
def initializeCTM(E,D, Chi0):
	D2 =D*D
	T = zz((D2, D2, D2))
	C = zz((D2, D2))
	
	E = np.reshape(E, (D*D,D*D,D*D,D,D))
	T = tsum('ijkll', E)	
	E = np.reshape(E, (D*D,D*D,D*D,D*D))
	C = np.reshape(T, (D*D,D*D,D,D))
	C = tsum('ijkk', C)
	Chi = D2
	
	while (Chi0)>Chi*D2:
		
		T2 = buildT2(T, E, Chi, D2)
		C2 = buildC2(C, T, T2, Chi, D2)
		T =T2
		C =C2
		Chi= Chi*D2
		
		print(C.shape, T.shape, Chi)
	return (C,T, Chi)
	

def buildT2(T, E, Chi, D2):			
	
#	E   = tsum('ijkl ->jkli',E)
	T2 = tsum('ijlk, lmn ', E, T)
	T2 = tsum('ijkmn -> ijmkn', T2)	
	T2 = T2.reshape(D2, D2, Chi, D2*Chi)
	
#	T21 = tsum('ijkl ->ilkj', T21)
	T2 = np.reshape(T2, (D2,D2*Chi, D2*Chi))
			
	T2 = T2/np.max(T2)	
	
	return T2
	
def buildC2(C, T, T2, Chi, D2):
				
	XD2 = Chi*D2
	Ctemp = tsum('ij, nim ->mnj', C, T)
	Ctemp = Ctemp.reshape(Chi,Chi*D2)
	
	C2 = zz((Chi*D2, Chi*D2))	
	T2   = tsum('ijk -> kij', T2)	
	C2 = tsum('ij, jkl', Ctemp, T2)

	C2 = tsum('ijk ->kji', C2)
	C2 = C2.reshape(Chi*D2, Chi*D2)
	
	return C2/np.max(C2)
	
	
	
def TruncateC2(C2, Chi, D2, cut):
	m = Chi*D2
	C2s = sp.csr_matrix(C2)
	#u, s, vh = np.linalg.svd(C2, full_matrices=True)
	#s,u = np.linalg.eig(C2)
	s, u = spla.eigs(C2s, k=min(cut+5,C2.shape[0]-2))#scipy.linalg.eig(C2, right = True)
	#vi = scipy.linalg.inv(u)
	count = 0
	trunc = 0
	cut0 = cut-1
	
	sort = np.argsort(np.abs(s))[:-len(s)-1:-1]
	s = s[sort]
	u = u[:,sort]
	#vi = vi[sort,:]
	
	while (trunc ==0 and cut0 <Chi*D2-1):		
		if s[cut0] ==0.:
			trunc = 1
		else:
			ratio = abs(s[cut0-1]/s[cut0])
			#if abs(abs(s[cut0-1])-abs(s[cut0])) > 10.**(-12):
			if abs(ratio -1.) > 10.**(-5):
				trunc = 1
			else:
				cut0 -= 1
			#print('cut0', s[cut0], s[cut0-1], cut0-1)
	#s = s
	sort = np.argsort(np.abs(s))[:-cut0-1:-1]
	u = u[:,sort]
	#vi= vi[sort,:]
	
	
	E = np.dot(np.transpose(u), u)
	
	E = sqrtm(E)
	E = scipy.linalg.inv(E)

	count = 0
	u = np.dot(u,E)
	
	Sig = np.diag(s[sort])
	LM  = u
	RM  = np.transpose(LM)#vi[sort,:]
	#Z   = np.eye(len(s[sort]))#np.dot(RM, LM)

	return (cut0, LM,Sig, RM)
	
def TruncateT2(T2, Chi, D2, m, LM, RM):
	n = Chi*D2
	T2temp1 = zz((D2,n,m))
	T2temp1 = tsum('ijl, kl ->ijk', T2, RM)
	Tt = tsum('ijk, jl ->ilk', T2temp1, LM)
	return Tt
				

def Hamiltonians(ds):
	Sp = zz((ds,ds))
	Sm = zz((ds,ds))
	Sz = zz((ds,ds))
	#dt = 0.001
	
	if ds == 5:
		for i in range(ds):
			Sz[i][i] = i - 2.
		Sm[0][1] = np.sqrt(4.)
		Sm[1][2] = np.sqrt(6.)
		Sm[2][3] = np.sqrt(6.)
		Sm[3][4] = np.sqrt(4.)
	
		Sp[1][0] = np.sqrt(4.)
		Sp[2][1] = np.sqrt(6.)
		Sp[3][2] = np.sqrt(6.)
		Sp[4][3] = np.sqrt(4.)	
	if ds ==2:
		Sz[0,0] = -0.5
		Sz[1,1] = 0.5
		Sm[0,1] = 1.
		Sp[1,0] = 1.
	S0  = np.eye(Sz.shape[0])
	
	Spm = tsum('ij,kl->ijkl', Sp,Sm)
	Spp = tsum('ij,kl->ijkl', Sp,Sp)
	Smp = tsum('ij,kl->ijkl', Sm,Sp)
	Smm = tsum('ij,kl->ijkl', Sm,Sm)
	Szz = tsum('ij,kl->ijkl', Sz,Sz)
	Sz1 = tsum('ij,kl->ijkl', Sz,S0)
	Sz2 = tsum('ij,kl->ijkl', S0,Sz)
	Sz  = Sz1+Sz2
	
	Sij = 0.5*(Spm + Smp) +Szz
	Sijp =  (-0.5*(Spp + Smm) -Szz).reshape((ds*ds, ds*ds ))#(0.5*(Spp + Smm) -Szz).reshape(ds*ds,ds*ds)
	
	
	Sij = np.reshape(Sij, (ds*ds, ds, ds))
	Sij = np.reshape(Sij, (ds*ds, ds*ds ))
	
	Szz = Szz.reshape(ds*ds,ds,ds)
	Szz = Szz.reshape(ds*ds,ds*ds)
	
	Sz  = np.reshape(Sz, (ds*ds, ds*ds))
		
	H = Sij#1./14.*(Sij + S2*7./10 + S3*7./45. + S4/90.)
			
	return Sij, Szz, Sijp

def Calculate2(Corner, T, E2Au,E2Bu, Chi, D2, ds, C1, C2):			

	#C1 = tsum('ij, abj -> iab', Corner, T)		
	
	#C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp1 = tsum('ijkl, omnkj-> oimnl', C2, E2Au)
	
	Ctemp1 = tsum('oijkl,ijm ->olkm', Ctemp1, C1)
	
	Ctemp2 = tsum('ijkl, omnkj-> oimnl', C2, E2Bu)
	
	Ctemp2 = tsum('oijkl,ijm ->olkm', Ctemp2, C1)

	Result = tsum('abcd, abcd', Ctemp1, Ctemp2)

	return Result
	
def Calculate2norm(Corner, T, E, Chi, D2):			

	C1 = tsum('ij, abj -> iab', Corner, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctemp0 = tsum('ijkl,ijm ->lkm', Ctemp0, C1)
	
	#Ctemp2 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	#Ctemp1 = tsum('ijkl,ijm ->lkm', Ctemp1, C1)
	
	Result = tsum('bcd, bcd', Ctemp0, Ctemp0)

	return Result, C1, C2
		
def Calculatenormcomplete(E,C1, C2):

	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctemp0 = tsum('ijkl,ijm ->lkm', Ctemp0, C1)
		
	return tsum('bcd, bcd', Ctemp0, Ctemp0)

def Calculatenormcomplete2(E0,E1,C1, C2):

	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E0)
	
	Ctemp0 = tsum('ijkl,ijm ->lkm', Ctemp0, C1)
		
	Ctemp1 = tsum('ijkl, mnkj-> imnl', C2, E1)
	
	Ctemp1 = tsum('ijkl,ijm ->lkm', Ctemp1, C1)

	return tsum('bcd, bcd', Ctemp0, Ctemp1)

					
def FullEnvironment(C1,C2, E):
	
	Ctemp0 = tsum('ijkl,kmnj ->inml', C2, E)
	
	Ctemp0 = tsum('ijkl,imno -> lkjmno', C2, Ctemp0)
	
	Ctemp0 = np.eisum('ijklmn, iopqmn -> jkl')
	
def EnvLayer(C1,C2,E22):
	
	Ctemp0 = tsum('ijkl,abkmnj ->abinml', C2, E22)
					
def GTensor(C1,C2, T, A, EB,D):
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, EB)

	Ctemp0 = tsum('ijkl,ijm ->lkm', Ctemp0, C1)
	
	Ctemp0 = tsum('lkm, abcl ->mkcba', Ctemp0, C2)
	
	Ctemp0 = tsum('abcde, efa ->fdcb', Ctemp0, C1)
	
	Ctemp0 = Ctemp0.reshape((D,D,D,D,D,D,D,D))
	
	Ctemp0 = tsum('abcdefgh ->acdgbdfh', Ctemp0)
	
	Ctemp0 = Ctemp0.reshape((D*D*D*D, D*D*D*D))
	
	return Ctemp0
					
def results(E2, Tensor1, Row1, Chi, D2):

	print('method0')
	if (Chi**2/float(D2**4)) < 1.:
		print('method1')
		Tensor0 = tsum('ijklmn, opmlkj ->iopn', Tensor1, E2)
				
		Result = 0.
		Result = tsum('abcd,abcd', Tensor0, Row1)
	else:
		print('method2')
		Tensor0 = tsum('abcd, ajklmd -> bcmlkj', Tensor1, Row1)
		Result = tsum('abcdef,abcdef', Tensor0, E2)
	
	print("End of calc", Result)
	return Result
				
def checkconv(spect01, spect11, tol):
	#Corner  = np.dot(Corner,Z1)
	Chi = len(spect01)
	m   = len(spect11)
	stdev = 0.
	for i in range(min(Chi,m)):
		stdev += (abs(spect01[i]) - abs(spect11[i]))**2
	stdev = np.sqrt(stdev)/min(Chi,m)
	if stdev < tol:
		return True, stdev
	else:
		return False, stdev
				
					
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

def declist(num, base, l):
     dig = np.array([0 for i in range(l)])
     quo = num
     div = base**(l-1)
     for i in range(l):             
             dig[i] = quo//div
             quo    = quo%div
             div = div//base
     return dig     

def gateH(H, ds,dt):
	H1 = tsum('ijkl ->ikjl', H.reshape(ds,ds,ds,ds)).reshape(ds*ds,ds*ds)
	exp1 = scipy.linalg.expm(1j*H1*dt)	
	
	return tsum('ijkl ->ikjl', exp1.reshape(ds,ds,ds,ds)).reshape(ds*ds,ds*ds)
	
	
def FullUpdate(FullE, V, W, Wp):
	W        = tsum('ijk ->ikj', W)
	Wp       = tsum('ijk ->ikj', Wp)
	
	FullE    = FullE.reshape(D**3, D**3, D**3, D**3)
	FullE1   = tsum('ijkl, mln -> mijkn', FullE , W)
	FullE1   = tsum('mijkn,oni -> mojk',  FullE1, V)
	S        = tsum('mojk, mkp -> opj',   FullE1, np.conjugate(Wp))
	
	FullE1   = tsum('ijkl, mln -> mijkn', FullE , Wp)
	R        = tsum('mijkn,mko -> nioj',  FullE1, np.conjugate(Wp))
	
	print(S.shape, R.shape)
	print (len(np.nonzero(np.round(S, decimals=5))[0])), (len(np.nonzero(np.round(R, decimals=5))[0]))
	
	Rmatrix       = R.reshape(D**4, D**4)
	Smatrix       = np.transpose(S.reshape(ds,D**4))
	Rin           = np.linalg.pinv(Rmatrix, rcon  =1e-10)
	Amatrix       = np.transpose(np.dot(Rin, Smatrix))
	Amatrix       = Amatrix/np.linalg.norm(Amatrix)
	print(np.linalg.norm(Rmatrix), np.linalg.norm(Smatrix), np.linalg.norm(Amatrix), 'norms')
	Vp            = Amatrix.reshape(ds,D, D**3)
	
	return Vp, R, S
	

	
def costfunction(FullE, V, W, Vp, Wp):
	t1 = np.kron(Vp, np.conjugate(Vp))
	t2 = np.kron(Wp, np.conjugate(Wp))
	print(t1.shape, t2.shape, '~')
	cost= tsum('ijk, ijk', t1, t2)
	return cost

def cost(FullE, R,S, Vp,Wp):
	term1 = tsum('ijk, ilm ->jklm', Vp,np.conjugate(Vp))
	print(term1.shape, R.shape, '+')
	term1 = tsum('ijkl, ijkl', R, term1)
	term2 = tsum('ijk, ijk', S, np.conjugate(Vp))
	term3 = tsum('ijk, ijk', np.conjugate(S), Vp)
	
	return term1 - term2 - term3
	
def SimpleUpdate(A,B,gg,ds,D):
	Au,As,Avt = np.linalg.svd(A)
	As0 = np.zeros(A.shape)
	for i in range(len(As)):
		As0[i,i] = As[i]
	Au  = np.dot(Au, As0).reshape(ds,D,D**3)
	
	Bu,Bs,Bvt = np.linalg.svd(B)
	Bs0 = np.zeros(A.shape)
	for i in range(len(Bs)):
		Bs0[i,i] = Bs[i]
	Bu  = np.dot(Bu, Bs0).reshape(ds,D,D**3)
	
	A2 = tsum('ijk,ljm ->ilmk', Au, Bu)
	A3 = tsum('ijkl, jlmn -> inkm', gg, A2)
	
	A2 = A3.reshape(ds*D**3, ds*D**3)
	A2u, A2s, A2vt = np.linalg.svd(A2)
	#print(A2s)
	
	sort = np.argsort(np.abs(A2s))[:-D-1:-1]
	#print(A2s[sort])
	A2s = np.diag(np.sqrt(A2s[sort]))
	A2u = np.dot(A2u[:,sort], A2s).reshape(ds,D**3,D)
	A2vt= np.transpose(np.dot(A2s,A2vt[sort,:])).reshape(ds,D**3,D)
	
	A2u = tsum('ijk->ikj', A2u)
	A2vt= tsum('ijk->ikj', A2vt)
	#print(A2vt.shape, A2u.shape)
	return A2u, A2vt

def TTensor(C1,C2,A1,B1,D,ds):
	E1  = tsum('acijkl, bcmnop -> abimjnkolp', A1,np.conjugate(A1))
	E1  = E1.reshape(ds*ds*ds*ds, D*D,D*D,D*D,D*D)
	
	E2  = tsum('acijkl, bcmnop -> abimjnkolp', B1,np.conjugate(B1))
	E2  = E2.reshape(ds*ds*ds*ds, D*D,D*D,D*D,D*D)
	
	T   = tsum('ijkl, amnkj ->aimnl', C2, E1)
	T   = tsum('aimnl, imj ->alnj', T,C1)
	
	T2  = tsum('ijkl, amnkj ->aimnl', C2, E2)
	T2  = tsum('aimnl, imj ->alnj', T2,C1)	
	
	T   = tsum('alnj, alnj', T, T2)
	
	return T

def RTensor(C1,C2,Bp,D,ds):
	
	E    = tsum('aijkl, amnop -> imjnkolp', Bp, np.conjugate(Bp))
	E    = np.reshape(E, (D*D,D*D,D*D,D*D))
	
	R    = tsum('ijkl, imn -> lkjmn', C2, C1)
	
	Temp = tsum('ijkl, mnkj ->imnl', C2, E)
	Temp = tsum('ijkl, ijm  -> lkm', Temp, C1)
	
	R    = tsum('ijklm, inm ->lnjk', R, Temp)
	R    = R.reshape(D,D, D,D, D,D, D,D)
	R    = tsum('ijklmnop -> ikmojlnp', R)
	R    = R.reshape(D*D*D*D, D*D*D*D)
	
	print('R :: ',R.shape, np.linalg.norm(R))
	
	return np.transpose(R)
	
def STensor(C1,C2,A1,B1,Bp,D,ds):	
	E    = tsum('abijkl, bmnop -> aimjnkolp', B1, np.conjugate(Bp))
	E    = np.reshape(E, (ds*ds, D*D,D*D,D*D,D*D))
	
	Temp = tsum('ijkl, imn -> lkjmn', C2, C1)
	
	S    = tsum('ijkl, amnkj -> aimnl', C2, E)
	S    = tsum('aijkl, ijm -> alkm', S, C1)
	
	S    = tsum('ijklm, ainm -> alnkj', Temp, S)
	S    = S.reshape(ds*ds, D,D, D,D, D,D, D,D)
	S    = tsum('aijklmnop -> aikmojlnp', S)
	S    = tsum('aijklmnop, abijkl-> bmnop', S, A1)
	S    = S.reshape(ds, D*D*D*D)
	
	return S

def OldCostfunction(R,S,Ap):
	E2 = tsum('ijklm, inopq -> jklmnopq', Ap, np.conjugate(Ap))
	E2 = E2.reshape(D*D*D*D, D*D*D*D)
	
	term1 = tsum('ij,ij', E2, R)
	A2 = Ap.reshape(ds, D*D*D*D)
	
	term2 = tsum('ij,ij', A2, S)
	
	Cost = term1 - term2 - np.conjugate(term2)
	print(Cost)
	return Cost
	
def iPEPSRecycle(E, Corner, TEdge, D, D2,Chi, Chimax, tol, Nsteps):
	inittime = time.time()
	cut = Chi+1
	#(Corner, TEdge, siz) = initializeCTM(E,D, Chi)

	#TEdge2  = buildT2(TEdge, E, siz, D2)#zz((D2, Chi*D2, Chi*D2))

	#Corner2 = buildC2(Corner, TEdge, TEdge2, siz, D2)#zz((Chi*D2, Chi*D2))
	step2 = time.time()
	
	#(m, LM,Corner,RM) = TruncateC2(Corner2, siz, D2, cut)
	#TEdge        = TruncateT2(TEdge2, siz, D2, m, LM, RM)
	#Chi = m
	
	step2 = time.time()
	print("Corner and Edge initialized")

	mx = np.max(abs(Corner))
	
	for isteps in range(Nsteps):

		cut = Chimax
		spect01 = np.diag(Corner)

		TEdge2  = buildT2(TEdge, E, Chi, D2)#zz((D2, Chi*D2, Chi*D2))
	
		Corner2 = buildC2(Corner, TEdge, TEdge2, Chi, D2)#zz((Chi*D2, Chi*D2))
		#print(Corner2)
		(m, LM,Corner,RM) = TruncateC2(Corner2, Chi, D2, cut)
		TEdge   = TruncateT2(TEdge2, Chi, D2, m, LM, RM)
				
		spectrum = np.diag(Corner)
		convergence, stdev =  checkconv(spect01, spectrum, tol)
		if convergence:
			totaltime = time.time()
			print( " - - -  CONVERGENCE SUCCESSFUL - - - ", totaltime - inittime, stdev)
			
			return (Corner, TEdge, Chi, isteps)	
			break
		
		print(isteps, stdev, Chi)
		Chi = m
		
	
def iPEPS(E, D, D2,Chi, Chimax, tol, Nsteps):
	inittime = time.time()
	cut = Chi+1
	(Corner, TEdge, siz) = initializeCTM(E,D, Chi)

	TEdge2  = buildT2(TEdge, E, siz, D2)#zz((D2, Chi*D2, Chi*D2))

	Corner2 = buildC2(Corner, TEdge, TEdge2, siz, D2)#zz((Chi*D2, Chi*D2))
	step2 = time.time()
	
	(m, LM,Corner,RM) = TruncateC2(Corner2, siz, D2, cut)
	TEdge        = TruncateT2(TEdge2, siz, D2, m, LM, RM)
	Chi = m

	step2 = time.time()
	print("Corner and Edge initialized")

	mx = np.max(abs(Corner))
	
	for isteps in range(Nsteps):

		cut = Chimax
		spect01 = np.diag(Corner)

		TEdge2  = buildT2(TEdge, E, Chi, D2)#zz((D2, Chi*D2, Chi*D2))
	
		Corner2 = buildC2(Corner, TEdge, TEdge2, Chi, D2)#zz((Chi*D2, Chi*D2))
		#print(Corner2)
		(m, LM,Corner,RM) = TruncateC2(Corner2, Chi, D2, cut)
		TEdge   = TruncateT2(TEdge2, Chi, D2, m, LM, RM)
				
		spectrum = np.diag(Corner)
		convergence, stdev =  checkconv(spect01, spectrum, tol)
		if convergence:
			totaltime = time.time()
			print( " - - -  CONVERGENCE SUCCESSFUL - - - ", totaltime - inittime, stdev)
			
			return (Corner, TEdge, Chi)	
			break
		
		print(isteps, stdev, Chi)
		Chi = m

def ReadTensors(ds,D,N, Nb):
	
	Tensors = zz((N, ds,D,D,D,D))
	for i0 in range(N):
		i = i0 +1
		string = 'Tensors/Tensor'+str(i)+'.dat'
		f = open(string,'r')
		f1 = f.readlines()
		#details = f1[0].split()
		print('Tensor' +str(i) +' is defined. file_len =', len(f1))
		f.close()
		for lines in f1:
			line = lines.split()
			Di, di1, di2, di3, di4, eps = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), eval(line[5])
			Tensors[i0, Di, di1, di2, di3, di4] = eps 
	
	Bonds = zz((Nb, D,D))

	for i0 in range(Nb):
		string = 'Tensors/BondOp'+str(i0+1)+'.dat'
		f = open(string,'r')
		f1 = f.readlines()
		#details = f1[0].split()
		print('Bond Operator -' +str(i0+1) +' is defined. file_len =', len(f1))
		f.close()
		for lines in f1:
			line = lines.split()
			Di, di1, di2,eps = int(line[0]), int(line[1]), int(line[2]), eval(line[3])
			Bonds[i0, di1, di2] = eps

	return Tensors, Bonds
	
def SaveEnv(Corner, TEdge, Chimax, tol):
		if os.path.isfile('Environment'):
			os.mkdir('Environment')
		f = open("Environment/Corner"+str(Chimax),'w+')
		for i in range(Corner.shape[0]):
			if abs(Corner[i,i].imag) > tol:
				img = Corner[i,i].imag
			else:
				img = Corner[i,i].imag#0.
				
			f.write(str(i)+'\t'+str(Corner[i,i].real/abs(Corner[0,0])) +'\t'+ str(img/abs(Corner[0,0]))+'\n')
		f.close()

		f = open("Environment/TEdge"+str(Chimax),'w+')
		for i in range(TEdge.shape[0]):
			for j in range(TEdge.shape[1]):
				for k in range(TEdge.shape[2]):
					if abs(TEdge[i,j,k]) >10.**(-14):
						strr = str(i)+'\t'+str(j)+'\t'+str(k)+'\t'+str(TEdge[i,j,k].real)
						if abs(TEdge[i,j,k].imag) > tol:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'
						else:
							strr = strr + '\t'+str(TEdge[i,j,k].imag)+'\n'#str(0.)+'\n'
						f.write(strr)
		f.close()

def GetEnv(E, Corner, TEdge, D, D2,Chi, Chimax, tol, Nsteps, Read, Recycle, Input, Saves):
	D2 = D*D
	if Input:
		(Corner, TEdge, Chi, steps) = iPEPSRecycle(E, Corner, TEdge, D, D2,Chi, Chimax, tol, Nsteps)
			
	
	if not Input and os.path.isfile('Environment/Corner'+str(Chimax)) and Read:
		print('Reading Corner and Edge')
		Corner = zz((Chimax, Chimax))
		TEdge  = zz((D2,Chimax, Chimax))
		f = open('Environment/Corner'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()
			index, realvalue, imagvalue = int(line[0]), float(line[1]), float(line[2])
			Corner[index, index] = complex(realvalue, imagvalue)
		f.close()
		f = open('Environment/TEdge'+str(Chimax),'r')	
		f1 = f.readlines()
		for lines in f1:
			line = lines.split()		
			i1, i2, i3, realvalue, imagvalue= int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
			TEdge[i1,i2,i3] = complex(realvalue, imagvalue)
		
		Chi = Chimax
	
		if Recycle == True:
			(Corner, TEdge, Chi, steps) = iPEPSRecycle(E, Corner, TEdge, D, D2,Chi, Chimax, tol, Nsteps)
			if steps >10 and Saves:
				SaveEnv(Corner, TEdge, Chimax,tol)
			print('****************** Recycled *****************')
	else:
		
		if not Input:
			print('iPEPS')
		
			(Corner, TEdge, Chi) = iPEPS(E,D, D2,Chi, Chimax, tol, Nsteps)
			if Saves:
				SaveEnv(Corner, TEdge, Chimax, tol)
	
			
	
	return (Corner, TEdge, Chi)

def Takagi(C):
	if np.allclose(C, C.T):
		msg = 'all ok'
		#print('Takagi valid')
	else:
		print('*****************Takagi failed*****************')
		sys.exit('Matrix is not symmetric')
	D,X = scipy.linalg.eig(C)
	sort = np.argsort(-np.abs(D))#[:-len(D)-1:-1]
	#print(D, sort, D[sort])
	D   = D[sort]
	X   = X[:,sort]
	Ein = X.T @ X
	E   = scipy.linalg.inv(Ein)
	#print('Takagi ::', np.round(E, decimals = 10))
	O   = X @ sqrtm(E) 
	
	if not np.allclose(C, O @ np.diag(D) @ O.T):
		Esq = sqrtm(E)
		Xin = scipy.linalg.inv(X)
		print('\n','*****************Takagi Failed*************','\n')
		sys.exit('TAKAGI INVERSION FAILED!!!!')
	return D, O

def renormers(Corner,TEdge,E):
	Chi = Corner.shape[0]
	D2  = E.shape[0]
	#dD  = E1.shape[0]
	
	siz = Chi
	cut = Chi+1
	
	TEdge2  = buildT2(TEdge, E, siz, D2)#zz((D2, Chi*D2, Chi*D2))

	Corner2 = buildC2(Corner, TEdge, TEdge2, siz, D2)#zz((Chi*D2, Chi*D2))
	step2 = time.time()
	
	(m, LM,Corner,RM) = TruncateC2(Corner2, siz, D2, cut)
	#TEdge        = TruncateT2(TEdge2, siz, D2, m, LM, RM)
	#Chi = m
	
	return LM, RM

def Ttemp(T, LM, RM):
	D2 = T.shape[0]
	Chi = T.shape[1]
	
	temp = tsum('aij, kim, njo -> kanmo', T, LM.reshape(D2,Chi, Chi), LM.reshape(D2,Chi, Chi))
	
	return temp
	
	
def TEdgenew(T, LM, RM, E1):
	Chi = T.shape[1]
	D2  = E1.shape[1]
	dD  = E1.shape[0]
	
	T2 = tsum('ijlk, lmn -> ijmkn', E1, T).reshape(dD, Chi*D2, Chi*D2)
	
	T2temp1 = tsum('ijl, kl ->ijk', T2/np.max(abs(T2)), RM)
	T = tsum('ijk, jl ->ilk', T2temp1, LM)
	
	return T#/np.max(C)


def FullUEnv(C,T,A, A1):
	ds = A.shape[0]
	Ddk0=A.shape[1]
	Ddk= A1.shape[1]
	D  = A.shape[2]
	D2 = D*D
	
	E  = EAcustom(A, A1)#tsum('aijkl, amnop -> imjnkolp', A, np.conjugate(A1)).reshape(Ddk0*Ddk,D2,D2,D2)
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctop   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
	
	#print(Ctop.shape, Ctemp0.shape, Ddk)
	Rho    = tsum('iaj, ibj  -> ab',  Ctop, Ctop)#.reshape(ds*D*ds*D, ds*D*ds*D)	
	
	return Rho.reshape(Ddk0,Ddk,Ddk0,Ddk)
	#Ctemp0 = tsum('aijkl,ijm -> alkm', Ctemp0, C1)

def FullUEnv_H(C,T,A):
	ds = A.shape[0]
	Ddk= A.shape[1]
	D  = A.shape[2]
	D2 = D*D
	
	E  = tsum('aijkl, amnop -> imjnkolp', A, np.conjugate(A)).reshape(Ddk*Ddk,D2,D2,D2)
	
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	Ctemp0 = tsum('ijkl, mnkj-> imnl', C2, E)
	
	Ctop   = tsum('ijkl, lkm -> ijm', Ctemp0, C1)
	
	#print(Ctop.shape, Ctemp0.shape, Ddk)
	Rho    = tsum('iaj, ibj  -> ab',  Ctop, Ctop)#.reshape(ds*D*ds*D, ds*D*ds*D)	
	
	return Rho
	
	
def XfromA(A, Tensors):
	N = Tensors.shape[0]
	X = np.array([0.,]*2*N)
	for i in range(N):
		j = 2*i
		unit = Tensors[i,:,:,:,:,:]
		dot1 = tsum('aijkl, aijkl', A, unit)
		dot2 = tsum('aijkl, aijkl', unit, unit)
		amp  = (dot1/dot2)
		X[j] = amp.real
		X[j+1] = amp.imag
	
	#print(np.round(X, decimals = 14))
	return X/np.max(abs(X))# np.round(X, decimals = 14)

def lmdfromX(X):
	X   = np.array(X)
	l =  len(X)
	X1 = zz((l//2))
	for i in range(l//2):
		X1[i] = X[2*i] + X[2*i + 1]*1j
		
	return X1
	
def Afromlambda(X, Tensors):
	
	lmd = lmdfromX(X)
	Ai    = tsum('i,ijklmn -> jklmn', lmd, Tensors)		
	return Ai
	
	
def Proj(Rhos, D, dk, ds):
	dk = Rhos.shape[0]
	D  = Rhos.shape[1]
	Rho = Rhos.reshape(dk*D,dk*D, dk*D,dk*D) # A At B Bt
	
	RhoT0 = np.einsum('ijkj->ik', Rho)
	
	RhoT  = RhoT0
	Di, Ui  = Takagi(RhoT)
	#Di   = np.random.random(Di.shape); Di = Di/np.max(abs(Di))
	sort = np.argsort(np.abs(Di))[:-D-1:-1]
	Di = Di[sort]; Ui = Ui[:,sort]
	
	Urn = np.random.random(RhoT.shape)
	Drn, Urn = Takagi(Urn + Urn.T)
	Rhon =  Ui.T @ RhoT @ Ui 
	P     = Ui @ Ui.T
	#P    = Urn @ Urn.T#Ui @ Ui.T
	#P    = np.random.random(RhoT.shape); P = P + P.T#np.eye(P.shape[0])#
	
	#print(np.round(Ui, 13), '~')
	
	check0 = np.trace(Rhon)
	print(check0)
	for i in range(20):
		RhoT= np.einsum('ijkl, jl -> ik', Rho, np.conjugate(P))
	
		Di, Ui0  = Takagi(RhoT)
		sort = np.arange(D); Di = Di[sort]; Ui = Ui0[:,sort]
		
		Rhon =  Ui.T @ RhoT @ Ui 
		P = Ui @ Ui.T
		
		check1 = np.trace(Rhon)
		dev    = np.abs(1 - check1/check0)
		
	#	print('~', i, dev, check0, check1)
		check0 = check1
			
		if dev < 1e-14:
	#		print(i,dev, check1)
			break

	return Ui0
	
def purif(Rho0):
		
	Ddk = Rho0.shape[0]
	Rho0 = np.round(Rho0, 14)
	RhoT  = Rho0.reshape(Ddk*Ddk, Ddk*Ddk)#tsum('aiaj', Rho0)
	
	RhoA   = tsum('ijaa', Rho0)
	
	Rd, Rs = Takagi(RhoT)
	Rs1 = (Rs.reshape(Ddk, Ddk, Ddk, Ddk))
	Rs1 = tsum('ijkl -> ikjl', Rs1).reshape(Ddk*Ddk, Ddk*Ddk)
	
	Rss, Rsu = scipy.linalg.eig(Rs1)
	
	#print('-', Rss)
	
	Rtak = np.eye(Ddk)
	RhoTnew = tsum('ikjl ,ia,  kc, jb,  ld ->acbd', Rho0, Rtak, np.conjugate(Rtak), Rtak,  np.conjugate(Rtak))
	
	#print('###########################')

	return RhoTnew, Rtak #@ U

def CTEnvironment(C,T):
	C1 = tsum('ij, abj -> iab', C, T)		
	
	C2 = tsum('abc, iaj-> jibc', C1, T)
	
	C3 = tsum('ijkl,imn -> lkjmn', C2, C1)
	
	return C3 #tsum('ijklm, inopm -> jnoplk', C3,C3 )
	
	
def Overlap2(Env,A,A1):
	
	E1 = EAcustom(A,A1)
	#E2 = EAcustom(A2,A1)
	E2 = tsum('ijkl-> klij', E1)
	
	#print(E1.shape, E2.shape, Env.shape, A.shape )
	return tsum('ijkl, mnip, lpmnjk', E1, E2, Env)
	


def Fidelity(X, Var):
	[Env,A1, Tensors] = Var
	#print(X.shape, '~')
	A  = Afromlambda(X,Tensors)
	
	
	term1 = Overlap2(Env,A,A1)
	term3 = Overlap2(Env,A,A)
	term4 = Overlap2(Env,A1,A1)
	
	#print(1 -np.abs(term1)**2/(term3*term4))
	return 1 -np.abs(term1)**2/(term3*term4)

def Fidelity0(A, Var):
	[Env,A1, Tensors] = Var
	
	term1 = Overlap2(Env,A,A1)
	term3 = Overlap2(Env,A,A)
	term4 = Overlap2(Env,A1,A1)
	return 1 -np.abs(term1)**2/(term3*term4)


def Fidelity4(U, Var):
	[Corner0,T2,A,A1, A4] = Var
	ds = A.shape[0]
	D  = A.shape[1]
	Ddk= A4.shape[1]
	
	
	A1u   = tsum('aijkl, ip -> apjkl', A1, U)
	A4u   = tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, U,U,U,U)
	
	Enum  = EAcustom(A1u, A1)
	Tnum  = tsum('abcd, bcdij -> aij', Enum, T2)
	Rhonum = FullUEnv(Corner0,Tnum,A4u,A4).reshape(D,Ddk,D,Ddk)
	term1 = tsum('ijij', Rhonum)
	
	Eden1 = EAcustom(A1u, A1u)
	Tden1  = tsum('abcd, bcdij -> aij', Eden1, T2)
	Rhoden = FullUEnv(Corner0,Tden1,A4u,A4u).reshape(D,D,D,D)
	term3 = tsum('ijij', Rhoden)
	
	
	Eden2 = EAcustom(A, A)
	Tden2  = tsum('abcd, bcdij -> aij', Eden2, T2)
	Rhoden = FullUEnv(Corner0,Tden2,A,A).reshape(D,D,D,D)
	term4 = tsum('ijij', Rhoden)
	
	
	return 1 -np.abs(term1)**2/(term3*term4)

def inguess(X, val):
	X1 = np.zeros(X.shape)
	#Xt = np.max(X)*val#*np.random.random()*0.2
	X1 = [i*(1 - val*np.random.random()) for i in X]
	return X1
	
def Optimise(X0, Var):
	
	F = Fidelity(X0,Var)
	X0 = inguess(X0, 1e-1)
	options = {'gtol':10.**(-7)}
	
	Result = opt.minimize(Fidelity, X0, args = (Var,), method = 'CG', options = options)#; Result = Result.x
	return Result.x

	
def Rotmatrix(RhoA):
	D = RhoA.shape[0]
	di = np.diag(RhoA)
	print(di)
	sort = np.argsort(-np.abs(di))
	
	if np.allclose((di[sort])[0], (di[sort])[1]):
		diff = sort[2]
	else:
		diff =sort[0]
	Rot = np.eye(D)# np.zeros((D,D));
	Rot[:,[0, diff]] = Rot[:,[diff, 0]]
	print(di, Rot)
	return Rot
	

def OptXval(RhoAval, RhoA, U):
	
	RhoAnew = tsum('ac, ai, ck-> ik', RhoAval, U, U)
	RhoAval, RhoAu = scipy.linalg.eig(RhoAval)
	RhoAval = np.diag(RhoAval)
	print(RhoAnew)
	return np.eye(RhoA.shape[0])

def A4toA(Corner, T2, A1, A, A4, U, Rot):
	ds = A.shape[0]
	D = A.shape[1]
	Ddk = A1.shape[1]
	dk  = Ddk//D
	sortD= np.arange(D)
	
	E1 = EAcustom(A,A1)
	
	E4 = EAcustom(A,A4)
	
	Urn = np.random.random((Ddk, Ddk))
	Drn, Urn = Takagi(Urn + Urn.T)
	
	Ui = U#Urn[:,sortD]#np.eye(Ddk,D)
	
	
	check0 = 1.
	for i in range(20):
		
		A4i = tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, Ui,Ui,Ui,Ui)
		A1i = tsum('aijkl, ip -> apjkl', A1, Ui)
		A3i = tsum('aijkl, jq, kr, ls -> aiqrs', A4,Ui,Ui,Ui)
		E1i = EAcustom(A1i, A1)
		print(T2.shape, E1i.shape, A3i.shape)
		T  = tsum('abcd, bcdij -> aij', E1i, T2)
		Rho  = FullUEnv(Corner,T,A3i, A4).reshape(dk*D,dk*D,dk*D,dk*D)
		
		Rhos = Rho.reshape(dk,D,dk,D,dk,D,dk,D)
		norm = tsum('iajbiajb', Rhos)
		Rhos = Rhos/norm
		RhoT = Rhos.reshape(dk*D, dk*D, dk*D, dk*D)
		
		Ui0   = Proj(Rhos.reshape(dk,D,dk,D,dk,D,dk,D), D, dk, ds) 
		Ui   = Ui0[:,sortD] @ Rot
		RhoTnew = tsum('ijkj, ip, kq', RhoT,Ui, Ui)
		check1 = np.trace(RhoTnew)
		
		dev = np.abs(1 - check1/check0)
		
		
		print(i, check0, check1, Ui.shape, dev)
		if dev < 1e-14:
			print(dev, check0,'----------')
			break
		check0 = check1
	
	X = np.eye(D)
	"""
	E11 = EAcustom(A1i, A1i)
	T1  = tsum('abcd, bcdij -> aij', E11, T2)
	A41 =  tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, Ui,Ui,Ui,Ui)
	Rho  = FullUEnv(Corner,T1,A41, A41)#.reshape(D,D,D,D)
	
	E00 = EAcustom(A, A1)
	T00  = tsum('abcd, bcdij -> aij', E00, T2)
	#A41 =  tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, Ui,Ui,Ui,Ui)
	Rho0  = FullUEnv(Corner,T00,A, A4)#.reshape(D,D,D,D)
	RhoA  = tsum('ijkj', Rho0)/tsum('ijij', Rho0)
	RhoAnew = RhoTnew/tsum('ii', RhoTnew)
	
	#X  = sqrtm(RhoAnew @ scipy.linalg.inv(RhoA))
	
	#print(np.round(X,14))
	
	
	num = tsum('ijij', Rho)
	den2= tsum('ijij', Rho0)
	#print(num, den, den2, 1 - num*np.conjugate(num)/(den*den2), Fidelity4(Ui, [Corner,T2,A,A1, A4] ))
	"""
	return Ui, X	


def X4fromA(Env0, A, A1u, A1):
	D  = A.shape[1]
	E1 = EAcustom(A1u, A1)
	E2 = tsum('ijkl, iomn -> lomnjk', E1,E1)
	
	E10 = EAcustom(A, A1)
	E20 = tsum('ijkl, iomn -> lomnjk', E10,E10)
	
	print(E2.shape, Env0.shape)
	Rl  = tsum('ijklmn, ijklmo', E2, Env0)
	Rl0 = tsum('ijklmn, ijklmo', E20, Env0)
	Rl  = tsum('ijkj', Rl.reshape(D,D,D,D))
	Rl0 = tsum('ijkj', Rl0.reshape(D,D,D,D))
	
	print(np.round(Rl,12))
	print(np.round(Rl0,12))
	Xl = np.diag(np.diag(Rl) / np.diag(Rl0))
	
	Ru  = tsum('ijklmn, ijklon', E2, Env0)
	Ru0  = tsum('ijklmn, ijklon', E20, Env0)
	Ru  = tsum('ijkj', Ru.reshape(D,D,D,D))
	Ru0 = tsum('ijkj', Ru0.reshape(D,D,D,D))
	
	Xu = np.diag(np.diag(Ru) / np.diag(Ru0))
	
	Ru  = tsum('ijklmn, ojklmn', E2, Env0)
	Ru0  = tsum('ijklmn, ojklmn', E20, Env0)
	Ru  = tsum('ijkj', Ru.reshape(D,D,D,D))
	Ru0 = tsum('ijkj', Ru0.reshape(D,D,D,D))
	
	Xd = np.diag(np.diag(Ru) / np.diag(Ru0))
	
	
	print(Xl)
	print(Xu)
	print(Xd)
	return Xl, Xu, Xd
	
def X4fromA1(Env, A,A1,A4, A1u, A4u, Corner, T2, X, U):
	
	#A1u = tsum('aijkl, ip -> apjkl', A1, U)
	#A4u =  tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, U,U,U,U)
	
	E1n = EAcustom(A1u, A1)
	Tn  = tsum('abcd, bcdij -> aij', E1n, T2)
	E4n = EAcustom(A4u, A4)
	
	E1d = EAcustom(A1u, A1u)
	Td  = tsum('abcd, bcdij -> aij', E1d, T2)
	E4d = EAcustom(A4u, A4u)
	
	E   = EAcustom(A,A)
	T0  = tsum('abcd, bcdij -> aij', E, T2)
	
	C1 = tsum('ij, abj -> iab', Corner, Tn)		
	C2 = tsum('abc, iaj-> jibc', C1, Tn)
	C3 = tsum('ijkl, im -> mjkl', C2, Corner)
	Envn = tsum('aijb, bkla -> lkji', C3, C3)
	num = tsum('ijkl, ijkl', Envn, E4n)
	
	C1 = tsum('ij, abj -> iab', Corner, Td)		
	C2 = tsum('abc, iaj-> jibc', C1, Td)
	C3 = tsum('ijkl, im -> mjkl', C2, Corner)
	Envd = tsum('aijb, bkla -> lkji', C3, C3)
	den = tsum('ijkl, ijkl', Envd, E4d)
	
	C1 = tsum('ij, abj -> iab', Corner, T0)		
	C2 = tsum('abc, iaj-> jibc', C1, T0)
	C3 = tsum('ijkl, im -> mjkl', C2, Corner)
	Envd2 = tsum('aijb, bkla -> lkji', C3, C3)
	
	den2 = tsum('ijkl, ijkl', Envd2, E)
	
	return(1 - num*np.conjugate(num)/(den*den2))
	

def A1toA(Corner, TEdge, A1):
		ds  = A1.shape[0]
		Ddk = A1.shape[1]
		D   = A1.shape[2]
		dk  = Ddk//D
		
		Rho  = FullUEnv_H(Corner,TEdge,A1).reshape(dk*D,dk*D,dk*D,dk*D)
		Rhos = Rho.reshape(dk,D,dk,D,dk,D,dk,D)
		norm = tsum('iajbiajb', Rhos)
		Rhos = Rhos/norm
		RhoT = Rhos.reshape(dk*D, dk*D, dk*D, dk*D)
		
		U0   = Proj(Rhos.reshape(dk,D,dk,D,dk,D,dk,D), D, dk, ds) 
		
		return  RhoT, U0

def XEnv(Corner, T, A, h2, U):
	ds  = A.shape[0]
	D   = A.shape[1]
	dk  = h2.shape[2]
	Chi = Corner.shape[0]
	A2  = tsum('aijkl, bapq -> bpiqjkl',A,h2).reshape(ds,D*dk, D*dk, D, D)
	
	A2u =  tsum('aijkl, ip, jq -> apqkl', A2, U, U)
	
	E2u = EAcustom(A2u,A2u) 
	R1  = tsum('ab, kbc, lad, ijkl ->dijc', Corner, T,T, E2u).reshape(Chi,D,D,D,D,Chi)
	
	R2  = tsum('aijklb, cmnolb -> aijkonmc', R1, R1)
	print(R2.shape)
	Rho = tsum('aijkonmc, apjqrsmc -> pqrsnoki', R2,R2)
	
	Rho1= tsum('pqqssoki -> ikpo',Rho).reshape(D*D, D*D)
	
	
	E20 = EAcustom(A,A) 
	R1  = tsum('ab, kbc, lad, ijkl ->dijc', Corner, T,T, E20).reshape(Chi,D,D,D,D,Chi)
	
	R2  = tsum('aijklb, cmnolb -> aijkonmc', R1, R1)
	print(R2.shape)
	Rho = tsum('aijkonmc, apjqrsmc -> pqrsnoki', R2,R2)
	
	Rho0= tsum('pqqssoki -> ikpo',Rho).reshape(D*D, D*D)
	
	print(np.trace(Rho1))
	print(np.trace(Rho0), np.trace(Rho1)/np.trace(Rho0))
	
	Rho1s, Rho1u = scipy.linalg.eig(Rho1)
	print(np.round(Rho1s, 12))
	
	Rho0s, Rho0u = scipy.linalg.eig(Rho0)
	print(np.round(Rho0s, 12))
	
	
		
	
	
	

def Hop(Corner0,TEdge0,Tensors, A0,H):
	
	ds = A0.shape[0]
	D  = A0.shape[1]
	Chi= Corner0.shape[0]
	D2 = D*D
	
	hsq= sqrtm(H)
	hd,ho = Takagi(H)
	dk = ds*ds
	dk2= dk#ds*ds
	Dd2= D*dk
	D2d2= Dd2*D
	
	E = EAcustom(A0,A0)
	print(hd)
	Gate_ij = tsum('ijkl->ikjl', H.reshape(ds,ds,ds,ds))
	
	sort = np.argsort(np.abs(hd))[:-dk-1:-1]
	sortD= np.arange(D)
	hd   = hd[sort]
	ho = ho[:, sort]
	
	htak = ho @ sqrtm(np.diag((hd)))#sqrtm(H)#ho @ sqrtm(np.diag((hd)))
	h1  = np.reshape(htak, (ds,ds,dk)) 
	
	sorti = np.arange(1)
	Hsimple = (ho[:,sorti] @ (np.diag(np.sqrt(hd[sorti])))).reshape(ds,ds,1)
	print(Hsimple.shape)
	#h1  = hsq.reshape(ds,ds,dk)
	print('ALL OK ::', np.allclose(H, tsum('ija, kla', h1, h1).reshape(H.shape)))
	
	h2   = tsum(' kib,ija -> kjba', h1, h1)
	h3   = tsum('abij, bck->acijk',h2, h1)
	h3   = (h3 + h3.transpose(0,1,3,4,2) + h3.transpose(0,1,4,2,3))/3.
	
	h4   = tsum('kjdc, jiba -> kidcba', h2, h2)
	h2   = (h2 + h2.transpose(0,1,3,2))/2.
	h4   = (h4 + h4.transpose(0,1,3,4,5,2) + h4.transpose(0,1,4,5,2,3) + h4.transpose(0,1,5,2,3,4))/4.
	
	Xin = XfromA(A0, Tensors); Xin = Xin/np.max(Xin)
	#A = Afromlambda(X,Tensors)
	A  = A0
	print('Xinitial ::', Xin)
	Rot = np.zeros((D,D)); Rot[0,1] = 1.; Rot[1,0] = 1.; Rot[2,2] = 1.
	
	C3 = CTEnvironment(Corner0,TEdge0)
	Env0 = tsum('ainmb, ajklb -> ijklmn', C3, C3)
	
	LM, RM = renormers(Corner0,TEdge0,E)
	T2     = Ttemp(TEdge0, LM, RM)
	
	X11    = Xin
	X4     = Xin
	
	for i in range(1):
	
		X = XfromA(A, Tensors); X = X/np.max(X)
		Asimple = tsum('yxk, xabcd -> ykabcd', Hsimple,A).reshape(ds, D,D,D,D) 
		A1   = tsum('yxk, xabcd -> ykabcd', h1,A).reshape(ds, dk*D,D,D,D)
		A4    = tsum('yxijkl, xabcd -> yiajbkcld', h4,A).reshape(ds,Dd2, Dd2, Dd2, Dd2)
		E1    = EAcustom(A1, A)
		
		
	###############
		
		
		X = XfromA(A, Tensors); X = X/np.max(X)
		A1   = tsum('yxk, xabcd -> ykabcd', h1,A).reshape(ds, dk*D,D,D,D)
		A4    = tsum('yxijkl, xabcd -> yiajbkcld', h4,A).reshape(ds,Dd2, Dd2, Dd2, Dd2)
		E1    = EAcustom(A1, A)
			
	###############
		
		RhoEnv = FullUEnv(Corner0,TEdge0,A,A1)
		normEnv = tsum('ijij', RhoEnv)
		RhoEnv = RhoEnv/normEnv
		RhoA   = tsum('ijkj', RhoEnv)
		RhoA   = RhoA/tsum('ii', RhoA)
		
		Rot  = Rotmatrix(RhoA)
		
		RhoT, U0 = A1toA(Corner0, TEdge0, A1)
		U    = U0[:,sortD] @ Rot
		
		U2, X0   = A4toA(Corner0, T2, A1, A, A4, U, Rot)
		
		Rhonew= tsum('abcd, ai, bj, ck, dl -> ijkl', RhoT, U2, np.conjugate(U0), U2, np.conjugate(U0))
		
		RhoAnew = tsum('ijkj', Rhonew)#/normnew
		normnum  = tsum('ii', RhoAnew)
		#RhoAnew = RhoAnew/tsum('ii',RhoAnew)
		
		print(normnum, '~', np.allclose(U, U2))
		
		RhoAnew2 = tsum('abcb, ai, ck -> ik', RhoT, U2, U2)
		#print(np.round(RhoAnew, 14))
		print(np.round(RhoAnew2,15), '~~++~~')
		
		X  = np.diag(np.sqrt(np.diag(RhoAnew) / np.diag(RhoA)))#sqrtm(RhoAnew @ scipy.linalg.inv(RhoA))
		
		A1u = tsum('aijkl, ip -> apjkl', A1, U2)
		A11 = tsum('aijkl, ip -> apjkl', A, X)
		A41 =  tsum('aijkl, ip, jq, kr, ls -> apqrs', A, X,X,X,X)
		A4u = tsum('aijkl, ip, jq, kr, ls -> apqrs', A4, U2,U2,U2,U2)
		
		print(np.round(X, 15))
		
		#XEnv(Corner0, TEdge0, A, h2, U2)
		#A1x = tsum('aijkl, ip, jq, kr, ls -> apqrs', A, X,Xu,Xl,Xd)
		print(Fidelity0(A, [Env0,A1, Tensors]))
		print(Fidelity0(A1u, [Env0,A1, Tensors]))
		
		O1 = X4fromA1(Env0, A,A1,A4, A1u, A4u, Corner0, T2, X, U)
		O0 = X4fromA1(Env0, A,A1,A4, A11, A41, Corner0, T2, X, U)
		O00 = X4fromA1(Env0, A,A1,A4, A, A, Corner0, T2, X, U)
		
		print(Env0.shape, A11.shape, A41.shape)
		print(O1, O0, O00)
		X11 = XfromA(A4u, Tensors)
		X4  = XfromA(A41, Tensors)
		print(X11)
		print(X4)
		#Fid = Fidelity0(A11, [Env0,A1, Tensors])
		#Fid1 = Fidelity0(A11, [Env0,A1, Tensors])
		#print(Fid, Fid1)
		#X11 = 1.
		#X41 = 1.
	return X11, X4,A4u, O1
	#"""
