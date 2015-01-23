import numpy

def cll(  w, lPx_true, lPx_all ):
    return -sum( numpy.sum(lPx_true*w[:,None],axis=0) - \
    numpy.log( numpy.spacing(1) + numpy.sum( numpy.exp( numpy.sum(lPx_all*w[None,:,None],axis=1) ),axis=0) ) )


def mse( w, py_true, lPx_all ):  
    wg_lPx_all = (lPx_all*w[None,:,None]).sum(1)
    wg_lPx_all =  wg_lPx_all - wg_lPx_all.max(0)
    return -numpy.sum( (numpy.exp( wg_lPx_all -  \
    numpy.log( numpy.spacing(1) + numpy.sum( numpy.exp( wg_lPx_all ),axis=0) ) ) - py_true)**2 )
 

def rbc( alpha, lPx_true, lPx_all ):
    return -( ( numpy.log( numpy.spacing(1) + (numpy.exp( lPx_true )*alpha[None,:,None] ).sum(1)  / \
    ( ( numpy.exp( lPx_all )*alpha[None,:,None] ).sum(1)).sum(0) )).sum()  + ( numpy.log( numpy.spacing(1) + alpha) ).sum() )
    
    
def allr( w, lPx_all, cls, y, lamb ):
    lPx_pos = lPx_all[cls,:,y == (cls+1)].T
    lPx_neg = lPx_all[cls,:,y != (cls+1)].T
    
    P = ( numpy.log( numpy.spacing(1) + (numpy.exp( lPx_pos )*w[None,:,None] ).sum(1)  / \
    ( ( numpy.exp( lPx_all[:,:,y == (cls+1)] )*w[None,:,None] ).sum(1)).sum(0) )).sum() / (y == (cls+1)).sum(0)
    
    N = ( numpy.log( numpy.spacing(1) + (numpy.exp( lPx_neg )*w[None,:,None] ).sum(1)  / \
    ( ( numpy.exp( lPx_all[:,:,y != (cls+1)] )*w[None,:,None] ).sum(1)).sum(0) )).sum() / (y != (cls+1)).sum(0)
    return -( P-N + lamb*((w**2).sum())  )
    
    
    