import numpy as np
from scipy import *
import scipy.optimize as opt
from scipy.linalg import sqrtm, expm
from matplotlib import pyplot as plt
import os.path
import time
import tensorflow as tf

from ctmrgt import iPEPS, GetEnv, doubleE, Hop, EAcustom
from ctmrgt import EA, Calculate2norm, Calculate2, Hamiltonians, doubleH2, B_singlet
from ctmrgt import Calculatenormcomplete2, B_singlet, gateH,  Calculatenormcomplete


def zz(*args, **kwargs):
    kwargs.update(dtype=np.complex128)
    return np.zeros(*args, **kwargs)


def tsum(string, *args):
    #args = tuple([torch.tensor(np.array(a, dtype = np.complex128)) for a in args])

    return tf.einsum(string, *args).numpy()


def SplitTensor(A, ds, D):
    P = 0.

    A1 = tsum('aijkl ->ajikl', A).reshape(ds*D, D*D*D)
    u, s, vh = np.linalg.svd(A1)
    s1 = np.zeros((ds*D, D*D*D))
    for i in range(min(ds*D, D*D*D)):
        s1[i, i] = s[i]

    Vh = np.dot(s1, vh).reshape((ds*D, D, D, D))
    #Vh = tsum('ijkl ->jikl', Vh)

    u = u.reshape(ds, D, ds*D)

    return Vh, u


def FullEnv(C1, C2, D, ds):

    Ctemp0 = tsum('ijkl, imn ->nmjkl', C2, C1)

    return tsum('iabcj, idefj -> adefcb', Ctemp0, Ctemp0)


def FullGate(G, ds):
    gate = G
    g0 = sqrtm(G)
    #g1   = sqrtm(G)

    if np.allclose(g0, g0.T):
        print('gate symmetric. all OK')
    else:
        print('gate not symmetric. could cause error.')

    v = g0.reshape(ds*ds, ds, ds)
    u = g0.reshape(ds, ds, ds*ds)

    g1 = tsum('ija, bki ->kjab', u, v)
    gate = tsum('ijab, jkcd -> ikdacb', g1, g1)

    return gate


def ReadTensors(ds, D, N, Nb):

    Tensors = zz((8, ds, D, D, D, D))
    for i0 in range(2):
        i = i0 + 1
        string = 'Tensors/Tensor'+str(i)+'.dat'
        f = open(string, 'r')
        f1 = f.readlines()
        #details = f1[0].split()
        print('Tensor' + str(i) + ' is defined. file_len =', len(f1))
        f.close()
        for lines in f1:
            line = lines.split()
            Di, di1, di2, di3, di4, eps = int(line[0]), int(line[1]), int(
                line[2]), int(line[3]), int(line[4]), eval(line[5])
            #Tensors[i0, Di, di1, di2, di3, di4] = eps
            if abs(eps) > 10.**(-14):
                #eps = 1.
                if i0 == 0:
                    if di1 != 1:
                        Tensors[0, Di, di1, di2, di3, di4] = eps
                        print(Di, di1, di2, di3, di4)
                    if di2 != 1:
                        Tensors[1, Di, di1, di2, di3, di4] = eps
                    if di3 != 1:
                        Tensors[2, Di, di1, di2, di3, di4] = eps
                    if di4 != 1:
                        Tensors[3, Di, di1, di2, di3, di4] = eps
                if i0 == 1:
                    if di1 == 1:
                        Tensors[4, Di, di1, di2, di3, di4] = eps
                        print(Di, di1, di2, di3, di4)
                    if di2 == 1:
                        Tensors[5, Di, di1, di2, di3, di4] = eps

                    if di3 == 1:
                        Tensors[6, Di, di1, di2, di3, di4] = eps
                    if di4 == 1:
                        Tensors[7, Di, di1, di2, di3, di4] = eps
            #print(Di, di1, di2, di3, di4, eps)
    Bonds = zz((Nb, D, D))

    for i0 in range(Nb):
        string = 'Tensors/BondOp'+str(i0+1)+'.dat'
        f = open(string, 'r')
        f1 = f.readlines()
        #details = f1[0].split()
        print('Bond Operator -' + str(i0+1) +
              ' is defined. file_len =', len(f1))
        f.close()
        for lines in f1:
            line = lines.split()
            Di, di1, di2, eps = int(line[0]), int(
                line[1]), int(line[2]), eval(line[3])
            Bonds[i0, di1, di2] = eps

    return Tensors, Bonds


def lmdfromX(X):
    X = np.array(X)
    l = len(X)
    X1 = zz((l//2))
    for i in range(l//2):
        X1[i] = X[2*i] + X[2*i + 1]*1j

    return X1


def Afromlambda(X, Tensors, Bond, D, ds):

    lmd = lmdfromX(X)
    Ai = tsum('i,ijklmn -> jklmn', lmd, Tensors)
    return Ai


def Afromlambdanew(X, Tensors):
    #X   = np.array([X[0],]*4 + [X[1],]*4 +  [X[2],]*4 +  [X[3],]*4)
    lmd = lmdfromX(X)
    Ai = tsum('i,ijklmn -> jklmn', lmd, Tensors)
    return Ai


def A2B2fromlambda(X, Tensors, Bond, D, ds):

    # tsum('i,ijklmn -> jklmn', Tmv, Tensors)
    A2 = Afromlambda(X, Tensors, Bond, D, ds)
    B2 = B_singlet(A2, D, ds, Bond)
    B2 = tsum('aijkl -> ailkj', B2)
    return tsum('aijkl, bmnoj -> abimnokl', A2, B2)


def FidelityValue(X, Variable):  # A2B2, CTMEnv, D2):

    [Tensors, Env0, Env1, norm] = Variable
    #X = X + 0.1*np.random.random(X.shape)
    a = Afromlambdanew(X, Tensors)
    ds = a.shape[0]
    D = a.shape[1]
    ea = EA(a, a, D, ds)

    #eaat = EAcustom(A4,a)

    term3 = tsum('ijkl, ijkl', Env0, ea)

    term1 = tsum('aijkl, aijkl', Env1, np.conjugate(a))
    term2 = np.conjugate(term1)
    #print(EnvH.shape, eaat.shape, ea.shape, a.shape)
    #term1 = tsum('ijkl, ijkl', EnvH, np.conjugate(eaat))
    #print(term1, term2, term3, norm	)
    Fid = term1*term2/(term3*norm)  # term1*term2/(np.abs(term3))

    return Fid  # 1. - Fid


def Ranges(X):
    X1 = np.zeros(X.shape)
    Xt = np.max(X)*0.15
    X1 = [slice(X[i] - Xt, X[i] + Xt, Xt*2.) for i in range(len(X))]
    return tuple(X1)


def inguess(X, val):
    X1 = np.zeros(len(X))
    Xt = np.max(X)*val  # *np.random.random()*0.2
    X1 = [i - Xt*np.random.random()*(np.random.randint(2)*2 - 1.) for i in X]
    print(X1)
    return X1


def Jacobian(x, variable):
    n = len(x)
    dx = 1e-10
    func = FidelityValue(x, variable)
    jac = np.zeros((n, n))
    for j in range(n):  # through columns to allow for vector addition
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (FidelityValue(x_plus, variable) - func)/Dxj
    return jac


def SingleOptimisation(A, Corner, TEdge, gate, Tensors):
    #A    = Afromlambdanew(X, Tensors)

    X1, X4, A4, Fid = Hop(Corner, TEdge, Tensors, A, gate)

    return A4, Fid, Fid


def Overlap(X, Variable):

    [Tensors, Env0, Env1, EnvH, A4, norm] = Variable
    Anew = Afromlambdanew(X, Tensors)

    ds = Anew.shape[0]
    D = Anew.shape[1]

    Anew = Afromlambdanew(X, Tensors)
    Enew = EAcustom(Anew, Anew)
    Eop = EAcustom(A4, Anew)

    norm = norm  # .real
    norm2 = tsum('ijkl, ijkl', Env0, Enew)  # .real
    norm3 = tsum('ijkl, ijkl', EnvH, Eop)  # .real

    dot = tsum('aijkl, aijkl', Env1, np.conjugate(Anew))

    over = norm3/np.sqrt(norm2*norm)  # dot*np.conjugate(dot)/((norm2*norm))

    #print(norm, norm2, norm3, np.allclose(Enew, Eop))
    return (1. - over)


def EnCal(X, H, CTMEnv, Tensors, Bond, D2, ds):

    AB = A2B2fromlambda(X, Tensors, Bond, D, ds)
    H_ij = tsum('ijkl->ikjl', H.reshape(ds, ds, ds, ds))

    Ab2t = tsum('abijklmn, abcd ->cdijklmn', AB, H_ij)

    Eab2t = tsum('abijklmn, abopqrst -> iojpkqlrmsnt', Ab2t,
                 np.conjugate(AB)).reshape(D2, D2, D2, D2, D2, D2)

    A = Afromlambda(X, Tensors, Bond, D, ds)
    Eab = EA(A, A, D, ds)
    Eab2 = doubleE(Eab, Eab, D2)

    num = tsum('ijklmn, ijklmn', CTMEnv, Eab2t)
    den = tsum('ijklmn, ijklmn', CTMEnv, Eab2)
    print('En::', num/den)
    return num/den


def XfromA(A, Tensors):
    N = Tensors.shape[0]
    X = np.array([0., ]*2*N)
    for i in range(N):
        j = 2*i
        unit = Tensors[i, :, :, :, :, :]
        dot1 = tsum('aijkl, aijkl', A, unit)
        dot2 = tsum('aijkl, aijkl', unit, unit)
        amp = (dot1/dot2)
        X[j] = amp.real
        X[j+1] = amp.imag
    return np.round(X, decimals=14)


def Average(X):
    av1 = [X[2*i] for i in range(4)]
    av2 = [X[2*i+1] for i in range(4)]
    av3 = [X[2*i+8] for i in range(4)]
    av4 = [X[2*i+9] for i in range(4)]

    av = [np.mean(av1), np.mean(av2), np.mean(av3), np.mean(av4)]

    return av


def Errorval(A, Ax):
    w12 = np.abs(tsum('aijkl, aijkl', A, (Ax)))**2

    w11 = (tsum('aijkl, aijkl', Ax, (Ax)))
    w22 = (tsum('aijkl, aijkl', A, (A)))

    return 1 - np.sqrt(w12/(w11*w22))

# READ TENSOR FILES


ds, D = 2, 3
dn = 4
N = 2
Nb = 1
Model = 'HIsing'


########################################
# DEFINE THE PARAMETERS
#######################################

D2 = D*D
Chimax = 16
Chi = Chimax
cut = Chi+1
Nsteps = 5000
tau = 0.05
tol = 10.**(-14)


numsize = 1
sizes = [Chi]
###################
# INITIALIZE
##################
if 1:
    Tmv = np.array([1. + 0.1j, ]*4 + [0.1 + 0.01*1j, ]*4)

    Bmv = np.array([1.])  # np.random.random(Nb)
    print(Tmv)
    f = open("Details", 'a+')
    for i in range(N):
        f.write(' paramater' + str(i+1)+'\t'+str(Tmv[i*4]) + '\n')
    f.write('\n')
    for i in range(Nb):
        f.write('BondOp'+str(i+1)+' paramater = '+str(Bmv[i])+'\n')
    f.close()

    Tensors, Bonds = ReadTensors(ds, D, N, Nb)

    X = [1., 0., 0.1, 0.0]
    X0 = X
    X = [1, 0]*4 + [0.1, 0]*4

    Anew = Afromlambdanew(X, Tensors)
    A = Afromlambdanew(X, Tensors)

    #A    = tsum('i,ijklmn -> jklmn', Tmv, Tensors)
    Bond = tsum('i,ijk    -> jk', Bmv, Bonds).T

    print(Bond.T)
    #Bond = np.eye(Bond.shape[0])
    Bt = sqrtm(Bond)
    print(Bt.real)
    B = B_singlet(A, D, ds, Bt)
    #E0 = EA(A,A,D,ds)
    #E  = EA(B,B,D,ds)

    #print(np.allclose(Anew, A))
    E = EA(A, A, D, ds)

    # for i in range(numsize):
    #	Chimax = sizes[i]
    #	Chi = Chimax
    #	cut = Chi+1
    if 1:
        #(Corner, TEdge, Chi)=	GetEnv(E, D, D2,Chi, Chimax, tol, Nsteps, True, False, True)
        (Corner, TEdge, Chi) = GetEnv(E, E, E, D, D2, Chi,
                                      Chimax, tol, Nsteps, True, True, False, True)

        # Calculate2norm(Corner, TEdge, E, Chi, D2)
        norm, C1, C2 = Calculate2norm(Corner, TEdge, E, Chi, D2)

        CTMEnv = FullEnv(C1, C2, D, ds)/norm

    (Sij, SzSz, Sijp) = Hamiltonians(ds)  # ip,i; jp,j

    H = Sijp
    Hn = tsum('ijkl -> ikjl', H.reshape(ds, ds, ds, ds)).reshape(ds*ds, ds*ds)

    ""

    Rho = tsum('ijklmn, iamn, jklb -> ab', CTMEnv, E, E)

    Rho2 = tsum('ijklmn, iamn, jkla', CTMEnv, E, E)

    print(np.round(np.diag(Rho), 13), np.trace(Rho), norm, Rho2)

    dt = -1e-2
    gate = gateH(H, ds, dt)

    #Result = SingleOptimisation(X,Corner, TEdge, gate, Tensors)
    #print("Result :: ", Result)
    #X = Result/np.max(Result)

    ""
    # [0.05, 0.025, 0.01, 0.0075, 0.0050, 0.0025, 0.001]#[0.125 - i*0.005 for i in range(25)]
    dtau = [0.1,  0.01]
    # [1,2,10,10]#[3, 7, 10, 25, 50, 50, 50, 50]
    ncounts = [int(5./i)+1 for i in dtau]

    tot_time = 0.3
    dx = []

    SiSj = [[] for i in range(len(dtau))]
    Eng = [[] for i in range(len(dtau))]
    timing = [[] for i in range(len(dtau))]

    SiSj2 = [[] for i in range(len(dtau))]
    Eng2 = [[] for i in range(len(dtau))]
    timing2 = [[] for i in range(len(dtau))]

    Fidlist = []
    Fid2list = []
    Err = [[] for i in range(len(X0))]

    TotalError = []
    count = -1
    Ntau0 = 8
    tol = 1e-7
    times = 0.
    for dt0 in dtau:
        dt = dt0
        gate = gateH(H, ds, dt)

        count = count + 1
        Xnew = X
        Ntau = ncounts[count]  # Ntau0*(count +1)#int(tot_time/dt) + 1
        times = 0.
        Anew = A
        for i in range(Ntau):

            Anew, Fid, Fid2 = SingleOptimisation(
                Anew, Corner, TEdge, gate, Tensors)

            #Anew = Afromlambdanew(Xnew, Tensors)
            Enew = EAcustom(Anew, Anew)
            print(np.allclose(Anew, A))
            (Corner, TEdge, Chi) = GetEnv(Enew, Corner, TEdge, D,
                                          D2, Chi, Chimax, tol, Nsteps, False, True, True, False)
            # Calculate2norm(Corner, TEdge, E, Chi, D2)
            norm, C1, C2 = Calculate2norm(Corner, TEdge, Enew, Chi, D2)
            #CTMEnv          = FullEnv(C1,C2,D,ds)/norm

            #Xnew = SingleStep(CTMEnv, Corner, TEdge, Tensors, Bond,D2,ds, Xnew, gate)

            E2Au, E2Bu = doubleH2(Anew, Anew, D, ds, H)
            #print('new ::', Xnew)
            En = Calculate2(Corner, TEdge, E2Au, E2Bu,
                            Chi, D2, ds, C1, C2)/norm

            times = times + dt0

            #Anew = Afromlambdanew(Xnew, Tensors)
            dA = Errorval(A, Anew)

            SiSj[count].append(dA)
            Eng[count].append(En)
            timing[count].append(times)

            Fidlist.append(Fid)  # (-np.log(abs(1.-Fid))/np.log(10.))

            Fid2list.append(Fid)  # (-np.log(abs(1.-Fid2))/np.log(10.))

            f = open("Corr", 'a+')
            f.write(str(times) + '\t' + str(dt)+'\t' + str(En) + '\n')
            f.close()

            f = open("Echo", 'a+')
            f.write(str(times) + '\t' + str(dt)+'\t' + str(dA) + '\n')
            f.close()

        plt.figure(1)
        plt.title((r'Echo, $\chi =$' + str(Chimax)))
        plt.ylabel(r'Echo')

        plt.xlabel(r'$\t$')
        plt.yscale('log')
        plt.plot(timing[count], SiSj[count], '-x',
                 label=r'$\tau$ = '+str(np.round(dtau[count], 5)))
        #plt.plot(dtau, Fid2list, '-o', label = 'full-step')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('Etemp.png')
        gate = gateH(H, ds, -dt)

        f = open("SiSj", 'a+')
        f.write(str(dt0) + '\t'+str(En))
        f.write('\n')
        f.close()
        #Xnew = SingleOptimisation(X,Corner, TEdge, gate, Tensors)
        #SingleStep(CTMEnv, Corner, TEdge, Tensors, Bond,D2,ds, Xnew, gate)
        for i in range(Ntau):

            Anew, Fid, Fid2 = SingleOptimisation(
                Anew, Corner, TEdge, gate, Tensors)

            #Anew = Afromlambdanew(Xnew, Tensors)
            Enew = EAcustom(Anew, Anew)
            print(np.allclose(Anew, A), tau)
            (Corner, TEdge, Chi) = GetEnv(Enew, Corner, TEdge, D,
                                          D2, Chi, Chimax, tol, Nsteps, False, True, True, False)
            # Calculate2norm(Corner, TEdge, E, Chi, D2)
            norm, C1, C2 = Calculate2norm(Corner, TEdge, Enew, Chi, D2)
            CTMEnv = FullEnv(C1, C2, D, ds)/norm

            #Xnew = SingleStep(CTMEnv, Corner, TEdge, Tensors, Bond,D2,ds, Xnew, gate)

            E2Au, E2Bu = doubleH2(Anew, Anew, D, ds, H)
            #print('new ::', Xnew)
            En = Calculate2(Corner, TEdge, E2Au, E2Bu,
                            Chi, D2, ds, C1, C2)/norm

            times = times - dt0

            #Anew = Afromlambdanew(Xnew, Tensors)
            dA = Errorval(A, Anew)

            SiSj2[count].append(dA)
            Eng2[count].append(En)
            timing2[count].append(times)

            f = open("Corr", 'a+')
            f.write(str(times) + '\t' + str(dt)+'\t' + str(En) + '\n')
            f.close()

            f = open("Echo", 'a+')
            f.write(str(times) + '\t' + str(dt)+'\t' + str(dA) + '\n')
            f.close()
        print(Average(Xnew))
        # print(Average(X0))

        print(Average(np.array(Xnew)) - np.array(X0))
        Errrau = Average(np.array(Xnew)) - np.array(X0)
        #dx.append(Average(np.array(Xnew) - np.array(X0)))
        # print(Errrau)
        for par in range(len(X0)):
            if abs(X0[par]) > 1e-13:
                print('+', par, Errrau[par], X0[par], abs(Errrau[par]/X0[par]))
                Err[par].append(-np.log(abs(Errrau[par]/X0[par]))/np.log(10.))
            else:
                Err[par].append(-np.log(abs(Errrau[par]))/np.log(10.))

        Anew = Afromlambdanew(Xnew, Tensors)
        dA = Errorval(A, Anew)
        TotalError.append(dA)

        print('~~~~', Err[0], Err[1], Err[2], Err[3])

        # plt.figure(1)
        #plt.title((r'Error vs $\tau$'))
        # plt.ylabel(r'-log_{10}(E)')
        # plt.xlabel(r'$\tau$')
        # for i in range(len(dtau)):
        #	plt.plot(Err[i], '-x')
        #plt.legend(loc = 'best')
        # plt.savefig('Errortemp.png')

    f = open("Result", 'a+')
    for i in range(len(Xnew)):
        f.write(str(Xnew[i]) + '\t')
    f.write('\n')

    f = open("Errors", 'a+')
    for i in range(len(dtau)):
        f.write(str(dtau[i]) + '\t' + str(Chimax) + '\t')
        for j in range(len(Err)):
            f.write(str(Err[j][i]) + '\t')
        f.write('\n')

    f = open("Fidval", 'a+')
    for i in range(len(dtau)):
        f.write(str(dtau[i]) + '\t' + str(Fidlist[i]) + '\t'+str(Fid2list[i]))
        # for j in range(len(Err)):
        #	f.write(str(Err[j][i]) + '\t')
        f.write('\n')

    Labels = [r'$\lambda^{real}_1$', r'$\lambda^{imag}_1$',
              r'$\lambda^{real}_2$', r'$\lambda^{imag}_2$']

    plt.figure(4)
    plt.title((r'Error vs $\tau$, $\chi =$' + str(Chimax)))
    plt.ylabel(r'Error$')

    plt.xlabel(r'$\tau$')
    plt.yscale('log')
    # for i in range(len(Err)):
    plt.plot(dtau, TotalError, '-o')
    #plt.plot(dtau, Fid2list, '-o', label = 'full-step')
    plt.legend(loc='best')
    plt.savefig('Error.png')
    # plt.show()

    plt.figure(5)
    plt.title((r'Deviation(t), $\chi =$' + str(Chimax)))
    plt.ylabel(r'Deviation')

    plt.xlabel(r'$\t$')
    plt.yscale('log')
    iss = range(len(SiSj))  # [1,3,5,6]
    for i in iss:
        plt.plot(timing[i], SiSj[i], '-',
                 label=r'$\tau$ = '+str(np.round(dtau[i], 5)))
    #plt.plot(dtau, Fid2list, '-o', label = 'full-step')
    plt.legend(loc='best')
    plt.savefig('Et.png')
    plt.show()

    plt.figure(6)
    plt.title((r'$S.S$, $\chi =$' + str(Chimax)))
    plt.ylabel(r'$S.S$')

    plt.xlabel(r'$\t$')
    # plt.yscale('log')
    iss = range(len(SiSj))  # [1,3,5,6]
    for i in iss:
        xax = []
        yay = []
        for x in timing[i]:
            xax.append(x)
        for y in Eng[i]:
            yay.append(y)

        for x in timing2[i]:
            xax.append(x)
        for y in Eng2[i]:
            yay.append(y)

        plt.plot(xax, yay, '-o', label=r'$\tau$ = '+str(np.round(dtau[i], 5)))
        plt.plot([x], [y], 'v',  color='r')
        #plt.plot(timing2[i], SiSj2[i], '-o', color = clrs[i])
    #plt.plot(dtau, Fid2list, '-o', label = 'full-step')
    plt.legend(loc='best')
    plt.savefig('SiSj.png', bbox_inches='tight')
#

    # """
