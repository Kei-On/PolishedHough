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
