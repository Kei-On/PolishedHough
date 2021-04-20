import numpy as np

conc = np.concatenate

def a2l(arr,ele_dim):
    arr = np.array(arr)
    s = arr.shape
    if s[len(s)-1] != ele_dim:
        raise Exception()
    return np.reshape(arr,[-1,ele_dim])

def l2a(list,sample_arr):
    s1 = sample_arr.shape
    s2 = list.shape
    newshape = np.concatenate([s1[:len(s1)-1],s2[len(s2)-1:]])
    return np.reshape(list,newshape)

class ndMatrix:
    def __init__(self,A):
        self.data = np.reshape(A,[-1])
        self.shape = np.array(A.shape)
        self.multiplier = [np.prod(np.concatenate([self.shape[i+1:],np.array([1])])) for i in range(len(self.shape))]
        self.multiplier = np.array(self.multiplier,dtype = np.int64)

        self.vecM = lambda A: np.reshape(A,[np.prod(self.SHAPE['input shape']),1])
        self.vecN = lambda B: np.reshape(B,[np.prod(self.SHAPE['output shape']),1])

        self.devecM = lambda a: np.reshape(a,self.SHAPE['input shape'])
        self.devecN = lambda b: np.reshape(b,self.SHAPE['output shape'])

    def len(self):
        return len(self.data)

    def ij2k(self,ij_arr):
        ij_arr = np.array(ij_arr)
        ij_list = a2l(ij_arr,len(self.shape))
        mul = np.broadcast_to(self.multiplier,[ij_list.shape[0],self.multiplier.shape[0]])
        k_list = np.sum(ij_list * mul, axis = 1, dtype = np.int64, keepdims = True)
        k_arr = l2a(k_list,ij_arr)
        return k_arr

    def k2ij(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        ij_list = np.zeros([len(k_list),self.multiplier.shape[0]])
        for i,m in enumerate(self.multiplier):
            ij_list[:,i:i+1] = np.array(k_list / m, dtype = np.int64)
            k_list = k_list % m
        ij_arr = l2a(ij_list,k_arr)
        return ij_arr

    def print(self):
        return np.reshape(self.data,self.shape)
    
    def copy(self):
        return ndMatrix(self.print())

    def is_valid(self,ij_arr):
        return np.logical_and(np.array(ij_arr) < self.shape,0 <= np.array(ij_arr))

    def get_by_k_arr(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        Ak_list = self.data[k_list]
        Ak_arr = l2a(Ak_list,k_arr)
        return Ak_arr

    def set_by_k_arr(self,k_arr,x_arr):
        k_arr,x_arr = np.array(k_arr),np.array(x_arr)
        s = x_arr.shape
        k_list,x_list = a2l(k_arr,1),a2l(x_arr,s[len(s)-1])
        self.data[k_list] = x_list

    def get(self,ij_arr):
        ij_arr = np.array(ij_arr)
        valid_ij = self.is_valid(ij_arr)
        ij_arr = ij_arr * valid_ij
        k_arr = self.ij2k(ij_arr)
        return self.get_by_k_arr(k_arr) * np.prod(valid_ij, axis = len(valid_ij.shape)-1, keepdims=True)
        
    def set(self,ij_arr,x_arr):
        ij_arr,x_arr = np.array(ij_arr),np.array(x_arr)
        k_arr = self.ij2k(ij_arr)
        self.set_by_k_arr(k_arr,x_arr)

def ip(ndA,SHAPE):
    a = SHAPE['input left borders']
    b = SHAPE['input right borders']
    M = SHAPE['input shape']

    def pii(u_arr):
        u_arr = np.array(u_arr)
        u_list = a2l(u_arr,M.shape[0])
        a_list = np.broadcast_to(a,[u_list.shape[0],a.shape[0]])
        b_list = np.broadcast_to(b,[u_list.shape[0],b.shape[0]])
        M_list = np.broadcast_to(M,[u_list.shape[0],M.shape[0]])
        uu_list = (u_list-a_list)/(b_list-a_list)*M_list
        uu_arr = l2a(uu_list,u_arr)
        return uu_arr

    def alpha(u_arr):
        return ndA.get(np.floor(pii(u_arr)))
    return alpha

def ep(beta,SHAPE):
    c = SHAPE['output left borders']
    d = SHAPE['output right borders']
    N = SHAPE['output shape']

    def piinv(vv_arr):
        vv_arr = np.array(vv_arr)
        vv_list = a2l(vv_arr,N.shape[0])
        c_list = np.broadcast_to(c,[vv_list.shape[0],c.shape[0]])
        d_list = np.broadcast_to(d,[vv_list.shape[0],d.shape[0]])
        N_list = np.broadcast_to(N,[vv_list.shape[0],N.shape[0]])
        v_list = vv_list/N_list*(d_list-c_list) + c_list
        v_arr = l2a(v_list,vv_arr)
        return v_arr

    B = ndMatrix(np.zeros(N))
    vv_arr = B.k2ij(np.arange(B.len()).reshape([B.len(),1]))
    B.set(vv_arr,beta(piinv(vv_arr)))
    return B

def integral(f,t_arr,w_arr):
    y = f(t_arr)
    w = np.broadcast_to(w_arr,y.shape)
    return np.sum(f(t_arr)*w_arr,axis = len(y.shape) - len(w_arr.shape))

def divide(a_arr,b_arr):
    a_arr,b_arr = np.array(a_arr),np.array(b_arr)
    a_list,b_list = a2l(a_arr,1),a2l(b_arr,1)

    t = np.logical_and(a_list == 0,b_list == 0)
    c_list = a_list.copy()
    c_list[np.where(t)] = 0
    b_list[np.where(t)] = 1
    return l2a(c_list/b_list,a_arr)

class Hough:
    def __init__(self,SHAPE):
        self.SHAPE = SHAPE
        
        image = np.ones(self.SHAPE['input shape'])
        ndInd = ndMatrix(image)
        self.indicator = ip(ndInd,self.SHAPE)
        
    def vecM(self,A):
        return np.reshape(A,[np.prod(self.SHAPE['input shape']),1])
    
    def vecN(self,B):
        return np.reshape(B,[np.prod(self.SHAPE['output shape']),1])
    
    def devecM(self,a):
        return np.reshape(a,self.SHAPE['input shape'])
    
    def devecN(self,b):
        return np.reshape(b,self.SHAPE['output shape'])

    def h_phi(self,alpha):
        def beta(v_arr): 
            return divide(
                self.SHAPE['h* phi'](alpha,v_arr),
                self.SHAPE['h* phi'](self.indicator,v_arr))
        return beta

    def H_phi(self,ndA):
        return ep(self.h_phi(ip(ndA,self.SHAPE)),self.SHAPE)
    
    def apply(self,input):
        ndA = ndMatrix(input)
        ndB = self.H_phi(ndA)
        return ndB.print()
    
    def getH(self,filename):
        n = np.prod(self.SHAPE['input shape'])
        m = np.prod(self.SHAPE['output shape'])
        H = np.zeros([m,n])
        I = np.identity(n)

        for i in range(n):

            a = I[:,i]
            A = self.devecM(a)
            B = self.apply(A)
            b = self.vecN(B)
            H[:,i:i+1] = b

            print('%d/%d'%(i,n))
            self.H = H
            np.save(filename,H)
        self.H_inv = np.linalg.pinv(H)

    def multiplyH(self,A = None,B = None):
        if B is None:
            a = self.vecM(A)
            b = np.dot(self.H,a)
            return self.devecN(b)
        if A is None:
            b = self.vecN(B)
            a = np.dot(self.H_inv,b)
            return self.devecM(a)

    def loadH(self,filename):
        self.H = np.load(filename)
        self.H_inv = np.linalg.pinv(self.H)
        
LINE = {
    'input left borders': np.array([0,0]),
    'input right borders': np.array([1,1]),
    'input shape': np.array([28,28]),
    'output left borders': np.array([0, -np.sqrt(2)]),
    'output right borders': np.array([2*np.pi, np.sqrt(2)]),
    'output shape': np.array([128,128]),

    'sig inv': lambda x: -np.log(1/x-1),
    'D sig inv': lambda x: 1/(1/x-1)*(1/(x*x)),
}

n = 128
LINE['integral nodes'] = np.linspace(0,1,n+2)[1:n+1].reshape(n,1)
LINE['integral weights'] = np.broadcast_to(LINE['integral nodes'][0],[n,1])

def phi_line(v_arr,t_arr):
    v_arr,t_arr = np.array(v_arr),np.array(t_arr)
    v_list,t_list = a2l(v_arr,2),a2l(t_arr,1)
    m,n = len(v_list),len(t_list)

    v = np.broadcast_to(v_list,[n,m,2])
    v = np.transpose(v,[1,0,2])
    t = np.broadcast_to(t_list,[m,n,1])
    t = np.transpose(t,[0,1,2])

    ans = np.zeros(np.array([m,n,2]))

    ans[:,:,0] = v[:,:,1] * np.cos(v[:,:,0]) - LINE['sig inv'](t[:,:,0]) * np.sin(v[:,:,0])
    ans[:,:,1] = v[:,:,1] * np.sin(v[:,:,0]) + LINE['sig inv'](t[:,:,0]) * np.cos(v[:,:,0])

    sv,st = v_arr.shape,t_arr.shape
    return np.reshape(ans,conc([sv[:len(sv)-1],st[:len(st)-1],[2]]))

def h_star_phi_line(alpha,v_arr):
    v_arr = np.array(v_arr)
    def inted_f(t_arr):
        sv = v_arr.shape
        t = np.broadcast_to(t_arr,conc([sv[:len(sv)-1],t_arr.shape]))
        return alpha(phi_line(v_arr,t_arr)) * np.abs(LINE['D sig inv'](t))
    return integral(inted_f,LINE['integral nodes'],LINE['integral weights'])

LINE['phi'] = phi_line
LINE['h* phi'] = h_star_phi_line

CIRCLE = {
    'input left borders': np.array([0,0]),
    'input right borders': np.array([1,1]),
    'input shape': np.array([28,28]),
    'output left borders': np.array([0,0,0]),
    'output right borders': np.array([1,1,1]),
    'output shape': np.array([32,32,32]),
}

n = 128
CIRCLE['integral nodes'] = np.linspace(0,1,n+2)[1:n+1].reshape(n,1)
CIRCLE['integral weights'] = np.broadcast_to(CIRCLE['integral nodes'][0],[n,1])

def phi_circle(v_arr,t_arr):
    v_arr,t_arr = np.array(v_arr),np.array(t_arr)
    v_list,t_list = a2l(v_arr,3),a2l(t_arr,1)
    m,n = len(v_list),len(t_list)

    v = np.broadcast_to(v_list,[n,m,3])
    v = np.transpose(v,[1,0,2])
    t = np.broadcast_to(t_list,[m,n,1])
    t = np.transpose(t,[0,1,2])

    ans = np.zeros(np.array([m,n,2]))

    ans[:,:,0] = v[:,:,0] + v[:,:,2] * np.cos(2*np.pi*t[:,:,0])
    ans[:,:,1] = v[:,:,1] + v[:,:,2] * np.sin(2*np.pi*t[:,:,0])

    sv,st = v_arr.shape,t_arr.shape
    return np.reshape(ans,conc([sv[:len(sv)-1],st[:len(st)-1],[2]]))

def h_star_phi_circle(alpha,v_arr):
    v_arr = np.array(v_arr)
    v_list = a2l(v_arr,3)
    v2_list = v_list[:,2:3]
    v2_arr = l2a(v2_list,v_arr)

    def inted_f(t_arr):
        sv,st = v_arr.shape,t_arr.shape
        v2 = np.broadcast_to(v2_arr,conc([st[:len(st)-1],v2_arr.shape]))
        v2 = np.transpose(v2,[1,0,2])
        return alpha(phi_circle(v_arr,t_arr)) * np.abs(2*np.pi*v2)
    return integral(inted_f,CIRCLE['integral nodes'],CIRCLE['integral weights'])

CIRCLE['phi'] = phi_circle
CIRCLE['h* phi'] = h_star_phi_circle

CONV = {
    'input left borders': np.array([0,0]),
    'input right borders': np.array([1,1]),
    'input shape': np.array([28,28]),
    'output left borders': np.array([0,0,0,0]),
    'output right borders': np.array([1,1,1,1]),
    'output shape': np.array([28,28,28,28]),

    'sig inv': lambda x: -np.log(1/x-1),
    'D sig inv': lambda x: 1/(1/x-1)*(1/(x*x)),
}

n = 128
CONV['integral nodes'] = np.linspace(0,1,n+2)[1:n+1].reshape(n,1)
CONV['integral weights'] = np.broadcast_to(CONV['integral nodes'][0],[n,1])

def get_phi(f,SHAPE):
    outdim,indim = len(SHAPE['output shape']),len(SHAPE['input shape'])
    tdim = len(SHAPE['integral nodes'][0])
    def phi(v_arr,t_arr):
        v_arr,t_arr = np.array(v_arr),np.array(t_arr)
        v_list,t_list = a2l(v_arr,outdim),a2l(t_arr,tdim)
        m,n = len(v_list),len(t_list)

        v = np.broadcast_to(v_list,[n,m,outdim])
        v = np.transpose(v,[1,0,2])
        t = np.broadcast_to(t_list,[m,n,tdim])
        t = np.transpose(t,[0,1,2])

        ans = f(v,t,m,n)

        sv,st = v_arr.shape,t_arr.shape
        return np.reshape(ans,conc([sv[:len(sv)-1],st[:len(st)-1],[indim]]))
    return phi

def f_conv(v,t,m,n):
    ans = np.zeros(np.array([m,n,2]))
    ans[:,:,0] = (v[:,:,2] - v[:,:,0]) * t[:,:,0] + v[:,:,0]
    ans[:,:,1] = (v[:,:,3] - v[:,:,1]) * t[:,:,0] + v[:,:,1]
    return ans

def phi_conv(v_arr,t_arr):
    v_arr,t_arr = np.array(v_arr),np.array(t_arr)
    v_list,t_list = a2l(v_arr,4),a2l(t_arr,1)
    m,n = len(v_list),len(t_list)

    v = np.broadcast_to(v_list,[n,m,4])
    v = np.transpose(v,[1,0,2])
    t = np.broadcast_to(t_list,[m,n,1])
    t = np.transpose(t,[0,1,2])

    ans = np.zeros(np.array([m,n,2]))

    ans[:,:,0] = (v[:,:,2] - v[:,:,0]) * t[:,:,0] + v[:,:,0]
    ans[:,:,1] = (v[:,:,3] - v[:,:,1]) * t[:,:,0] + v[:,:,1]

    sv,st = v_arr.shape,t_arr.shape
    return np.reshape(ans,conc([sv[:len(sv)-1],st[:len(st)-1],[2]]))

def h_star_phi_conv(alpha,v_arr):
    v_arr = np.array(v_arr)
    v_list = a2l(v_arr,4)
    def inted_f(t_arr):
        sv,st = v_arr.shape,t_arr.shape
        v = []
        for i in (0,1,2,3):
            vi_list = v_list[:,i:i+1]
            vi_arr = l2a(vi_list,v_arr)
            v.append(np.broadcast_to(vi_arr,conc([st[:len(st)-1],vi_arr.shape])))
            v[i] = np.transpose(v[i],[1,0,2])
        g = np.sqrt(np.square(v[2]-v[0])+np.square(v[3]-v[1]))
        g[np.where(g==0)] = 1
        return alpha(CONV['phi'](v_arr,t_arr)) * g
    return integral(inted_f,CONV['integral nodes'],CONV['integral weights'])

CONV['phi'] = phi_conv
CONV['phi'] = get_phi(f_conv,CONV)
CONV['h* phi'] = h_star_phi_conv
