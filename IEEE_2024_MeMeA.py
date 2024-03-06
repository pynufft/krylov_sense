"""
Krylove subspace method
For coil sensitivity profiles
"""
import matplotlib.pyplot
import numpy as np
from numpy import pi
import pylab as plt
from colorsys import hls_to_rgb
import scipy.sparse.linalg
import matplotlib.pyplot
# from scipy.integrate.tests.test_integrate import pi
matplotlib.pyplot.gray()
# np.random.seed(123123)
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 0.5*r#1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)

    c = np.transpose(c, (1,2,0))
    return c


import numpy as numpy
import numpy
import scipy.sparse.linalg
import scipy.ndimage
import matplotlib


def get_norm_univector(u_bar):
    beta = numpy.linalg.norm(u_bar)
    u = u_bar / beta
    return beta, u

def create_fake_coils(N, n_coil):
    
    xx,yy = numpy.meshgrid(numpy.arange(0,N),numpy.arange(0,N))
    
    
#     
#     ZZ= numpy.exp(-((xx-128)/32)**2-((yy-128)/32)**2)     
    coil_sensitivity = numpy.zeros((N,N,n_coil)).astype(numpy.complex64)
    
    #image_sense = ()
    
    r = N/4
    phase_factor = 6
    for nn in range(0,n_coil):
        
        tmp_angle = nn*2*numpy.pi/n_coil
        shift_r = int(N/3)
        shift_x= (numpy.cos(tmp_angle)*shift_r).astype(numpy.complex64)
        shift_y= (numpy.sin(tmp_angle)*shift_r).astype(numpy.complex64)
#         ZZ= numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)
        ZZ=  numpy.exp(-phase_factor*numpy.random.randn(1)*1.0j*((xx-N/2-shift_x)*numpy.cos(180*nn/n_coil)/N + (yy-N/2-shift_y)*numpy.sin(180*nn/n_coil)/N))* numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)  
#         coil_sensitivity +=(numpy.roll(numpy.roll(ZZ,shift_x,axis=0),shift_y,axis=1),)
        coil_sensitivity[:,:,nn] = ZZ
            
    return coil_sensitivity

def _power(z, maxiter, type = 'random', scale=1.0):
    
    if type =='ones':
        v = numpy.ones_like(z, dtype=numpy.complex64)
    if type =='random':
        v = numpy.random.randn(*z.shape) + 1.0j*numpy.random.randn(*z.shape)
        
    # Normalize z  to avoid overflow
    z = z/numpy.linalg.norm(z.flatten()) 
    znorm = (numpy.einsum('ac,ac->', z.conj(), z)/scale)**0.5
    # znorm = numpy.linalg.norm(z.flatten())/scale**0.5
    for iter in range(0, maxiter):
        # V2 = numpy.einsum('abd, abc, abc -> abd', z, z.conj(), V)
        tmp = numpy.einsum('ac, ac -> a',  z.conj(), v)
        tmp = numpy.einsum('ad,a  -> ad', z, tmp)
        # Bv_norm = numpy.linalg.norm(z.conj() * V2)
        tmp = numpy.einsum('ac, ac -> ',  z.conj(), tmp)
        w = numpy.einsum('ad,  -> ad', z, tmp)/znorm**4
        # beta=numpy.linalg.norm(w.flatten(), ord=2,)
        beta = numpy.einsum('ac,ac->',w.conj(), w)**0.5
        w=w/beta
        # sigma = numpy.einsum('abc, abc-> ', V2.conj(), V2)**1
        # sigma = numpy.linalg.norm(w, ord=2, axis=2)
        sigma = numpy.einsum('ac,ac->a',w.conj(), w)**0.5
        # sigma.shape = z.shape[0:2] + (1,)
        # v = (w+1e-15)/(sigma+1e-15)
        v = numpy.einsum('ac,a->ac',w+1e-15, 1/(sigma+1e-15))


    tmp = numpy.sum(v.flatten())
    v = v/(tmp/abs(tmp))
    return v

def power(z, maxiter, type='random',scale=1.0):
    """
    Power iteration with a second iteration
    
    """
    image_shape = z.shape
    c = image_shape[-1]
    prod_geometry = numpy.prod(image_shape[:-1])
    z2 = z.copy()
    z2.shape = (prod_geometry, c)
    v = _power(z2, maxiter,'random', scale=scale)
    v.shape = image_shape

    return v

def lanczos92(z, maxiter, type='random',scale=1.0):
    """
    Lanczos iteration for generalized eigenvalue problem (GEP)
    """
    image_shape = z.shape
    c = image_shape[-1]
    prod_geometry = numpy.prod(image_shape[:-1])
    z2 = z.copy()
    z2.shape = (prod_geometry, c)
    v = _lanczos92(z2, maxiter,'random', scale=scale)
    v.shape = image_shape

    return v

def _lanczos92(z, maxiter, type='random',scale=1.0):
    # z = z*1.0
    if type =='ones':
        q = numpy.ones_like(z, dtype=numpy.complex64)
    if type =='random':
        q = numpy.random.randn(*z.shape) + 1.0j*numpy.random.randn(*z.shape)
    z = z/numpy.linalg.norm((z.flatten()))
    
    # znorm = numpy.linalg.norm(z.flatten())/numpy.prod(z.shape[0:2])/scale
    # znorm =numpy.einsum('abc,abc->',z,z.conj())/(numpy.prod(z.shape)**2)/scale
    znorm = 1/ (numpy.prod(z.shape)**2)/scale
    # V2 = numpy.einsum('abd, abc, abc -> abd', z, z.conj(), V)
    q_old = q*0.0
    beta = 0
    
    #normalize q along all directions
    q = q/numpy.einsum('ac,ac->',q.conj(),q)**0.5
    
     # normalize q along coil direction
    sigma = numpy.einsum('ac,ac->a',q.conj(), q)**0.5
    q = numpy.einsum('ac,a->ac',q+1e-15, 1/(sigma+1e-15))

    for iter in range(0, maxiter):
        
        
        # v = A q
        tmp = numpy.einsum('ac, ac -> a',  z.conj(), q)
        v = numpy.einsum('ac, a -> ac', z, tmp)
        
        # alpha = <q,v>
        alpha = numpy.einsum('ac,ac->',q.conj(), v)
        
        # w= B^{+} v
        tmp = numpy.einsum('ac, ac -> ',  z.conj(), v)
        w = numpy.einsum('ac,  -> ac', z, tmp)/znorm**2
        
        w = w - alpha*q - beta*q_old
        beta = numpy.einsum('ac,ac->',w.conj(), v)**0.5
        if beta < 1e-14:
            break
        w = w/beta
        q_old = q
        
        # normalize q along coil direction
        sigma = numpy.einsum('ac,ac->a',w.conj(), w)**0.5
        q = numpy.einsum('ac,a->ac',w+1e-15, 1/(sigma+1e-15))
        # print('k=', iter, 'alpha=', alpha, 'beta=',beta)
    tmp = numpy.sum(q.flatten())
    q = q/(tmp/abs(tmp))
    return q



def lanczos91(z, maxiter, type='random',scale=1e+4):
    """
    Lanczos iteration with the low-rank approximation
    Actually Lanczos iteration is a power iteration with orthogonalization. 
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjftcHYsbSEAxV2sFYBHW3qAhkQFnoECA0QAQ&url=https%3A%2F%2Fwww.cs.ucdavis.edu%2F~bai%2FWinter09%2Fkrylov.pdf&usg=AOvVaw3ybm2Pq0_4lYM1X0Hdsh07&opi=89978449
    """
    image_shape = z.shape
    c = image_shape[-1]
    prod_geometry = numpy.prod(image_shape[:-1])
    z2 = z.copy()
    z2.shape = (prod_geometry, c)
    v = _lanczos91(z2, maxiter,'random', scale=scale)
    v.shape = image_shape

    return v

def _lanczos91(z, maxiter, type='random',scale=1e+4):
    # z = z*1.0
    if type =='ones':
        v = numpy.ones_like(z, dtype=numpy.complex64)
    if type =='random':
        v = numpy.random.randn(*z.shape) + 1.0j*numpy.random.randn(*z.shape)
    # print('z.norm',numpy.linalg.norm(z.flatten()), )
    # z = z* 1e-5
    z = z/numpy.linalg.norm(z.flatten())#**2
    
    # B_norm of v
    q = v/numpy.einsum('ac,ac,ac,ac->',v.conj(),z,z.conj(),v)**0.5
    
    # normalize q along coil direction
    sigma = numpy.einsum('ac,ac->a',q.conj(), q)**0.5
    q = numpy.einsum('ac,a->ac',q+1e-15, 1/(sigma+1e-15))
   
    q_old = q*0
    beta = 0
    
    znorm = 1/(numpy.prod(z.shape)**2)/scale
   
    for iter in range(0, maxiter):
        
        
        tmp = numpy.einsum('ac, ac -> a ',  z.conj(), q)
        tmp = numpy.einsum('ac, a -> ac', z, tmp)
        
        tmp = numpy.einsum('ac, ac -> ',  z.conj(), tmp)
        w = numpy.einsum('ac,  -> ac', z, tmp)/znorm**2
        
        alpha = numpy.einsum('ac,ac,ac,ac->',q.conj(),z, z.conj(), w)
        
        w = w - alpha*q - beta*q_old
        
        beta = numpy.einsum('ac,ac,ac,ac->',w.conj(), z,z.conj(), w)**0.5
        if beta < 1e-14:
            break
        q_old = q
        w = w/beta
        
        sigma = numpy.einsum('ac,ac->a',w.conj(), w)**0.5
        q = numpy.einsum('ac,a->ac',w+1e-15, 1/(sigma+1e-15))
    tmp = numpy.sum(q.flatten())
    q = q/(tmp/abs(tmp))
    return q


# def get_norm_univector_tensor(u_bar):
#     beta = numpy.linalg.norm(u_bar)
#     u = u_bar / beta
#     return beta, u
# def lsmr_tensor(b, z, maxiter, threshold):
# #     def A(input):
# #         v_out = numpy.einsum('abd, abc, abc-> abd', z, z.conj(), input)
# #         return v_out
#     def A(x):
# #         k2 = numpy.fft.fftn( z.conj()*x, axes=(0,1))
# #         k2[1:,:] = 0
# #         k2[:, 1:] = 0
# #         out = numpy.fft.ifftn(k2, axes=(0,1))
#
#         k2 = numpy.einsum('abc, abc -> c',z.conj(), x)
#         out = numpy.einsum('abc, c -> abc', z, k2)
# #         out = numpy.einsum('mnc, abc, abc-> mnc',z,z.conj(), x)
# #         out *= z
# #         out = numpy.einsum('abd, abc, abc-> abd',z2,z2.conj(), x)
#         return out
#     beta, u = get_norm_univector_tensor(b)
#     alpha, v = get_norm_univector_tensor(A(u))
#     alpha_bar = alpha
#     zeta_bar = alpha*beta
#     rho = 1
#     rho_bar = 1
#     c_bar = 1
#     s_bar = 0
#     h = v
#     h_bar = 0
#     x = 0*v.copy()
#     k = 0
#     for k in range(0, maxiter):
# #     while numpy.linalg.norm(A(x) - b)/numpy.linalg.norm(b) > threshold:
# #         if k < maxiter:
#             beta, u = get_norm_univector_tensor(A(v) - alpha * u)
#             alpha, v = get_norm_univector_tensor(A(u) - beta * v)
#             rho_bar_old = rho_bar
#             rho_old = rho
#             rho = (alpha_bar**2 + beta**2)**0.5
#             c = alpha_bar / rho
#             s = beta / rho
#             theta = s*alpha
#             alpha_bar = c*alpha
#             theta_bar = s_bar * rho
#             rho_bar = ( (c_bar * rho )**2+ theta**2)**0.5
#             c_bar = c_bar * rho / rho_bar
#             s_bar = theta/rho_bar
#             zeta = c_bar * zeta_bar
#             zeta_bar = -s_bar * zeta_bar
#             h_bar = h - h_bar*theta_bar * rho / (rho_old*rho_bar_old)
#             x = x + h_bar*zeta/(rho*rho_bar)
#             h = v - h * theta / rho
# #             k+=1
# #             print('iterate: ',k, numpy.linalg.norm(A(x) - b)/numpy.linalg.norm(b))
# #         else:
# #             pass
#
#     return x
#     return d

def asymmetric_least_squares_tensor(b, z, maxiter):
    shape = b.shape
    def matvec(x): # x is a vector
        k2 = numpy.einsum('ac, ac -> c',z.conj(), x.reshape(shape))
        out = numpy.einsum('ac, c -> ac', z, k2)
        return out.flatten()
    def rmatvec(v):
        return matvec(v)
    A = scipy.sparse.linalg.LinearOperator((numpy.prod(shape),numpy.prod(shape)),matvec, rmatvec)
    
    x = scipy.sparse.linalg.lsqr(A, b.flatten(), iter_lim=maxiter)[0]
    
    
    return x.reshape(shape)

def solve_x(b, V, maxiter):

    def A(x):

        out = numpy.einsum('abc, ab -> abc', V, x)

        return out
    def AT(y):
        out = numpy.einsum('abc, abc -> ab', V.conj(), y)
        return out
    
    beta, u = get_norm_univector(b)
    alpha, v = get_norm_univector(AT(u))
    alpha_bar = alpha
    zeta_bar = alpha*beta
    rho = 1
    rho_bar = 1
    c_bar = 1
    s_bar = 0
    h = v
    h_bar = 0
    x = 0*v.copy()
    k = 0
    for k in range(0, maxiter):
#     while numpy.linalg.norm(A(x) - b)/numpy.linalg.norm(b) > threshold:
#         if k < maxiter:
            beta, u = get_norm_univector(A(v) - alpha * u)
            alpha, v = get_norm_univector(AT(u) - beta * v)
            rho_bar_old = rho_bar
            rho_old = rho
            rho = (alpha_bar**2 + beta**2)**0.5
            c = alpha_bar / rho
            s = beta / rho
            theta = s*alpha
            alpha_bar = c*alpha
            theta_bar = s_bar * rho
            rho_bar = ( (c_bar * rho )**2+ theta**2)**0.5
            c_bar = c_bar * rho / rho_bar
            s_bar = theta/rho_bar
            zeta = c_bar * zeta_bar
            zeta_bar = -s_bar * zeta_bar
            h_bar = h - h_bar*theta_bar * rho / (rho_old*rho_bar_old)
            x = x + h_bar*zeta/(rho*rho_bar)
            h = v - h * theta / rho

    return x

def inexact_inverse_iteration(z, maxiter, type='ones'):
    image_shape = z.shape
    c = image_shape[-1]
    prod_geometry = numpy.prod(image_shape[:-1])
    z2 = z.copy()
    z2.shape = (prod_geometry, c)
    v = _inexact_inverse_iteration(z2, maxiter,'random')
    v.shape = image_shape

    return v   

def _inexact_inverse_iteration(z, maxiter, type='ones'):

    def B(x):
        
        tmp = numpy.einsum('ac, ac-> a',z.conj(), x)
        out = numpy.einsum('ac,a -> ac',z, tmp)
        return out
    def A(x):
 
        k2 = numpy.einsum('ac, ac -> ',z.conj(), x)
        out = numpy.einsum('ac, -> ac', z, k2)

        return out

        
    if type =='ones':
        X = numpy.ones_like(z, dtype=numpy.complex64)
    if type =='random':
        X = numpy.random.randn(*z.shape) + 1.0j*numpy.random.randn(*z.shape)
    
    Y = numpy.zeros_like(X)
    r0 = B(X) - A(Y)
    for k in range(0, maxiter):
        r = B(X) - A(Y)
       
        d = asymmetric_least_squares_tensor(r, z, maxiter=4)
        Y += d
        
        X = Y/numpy.einsum('ac,ac->',Y.conj(), Y)**0.5
        
        sigma = numpy.linalg.norm(X, ord=2, axis=1)
        sigma.shape += (1,) 
        X = (X+1e-7)/(sigma+1e-7)
        # print(numpy.linalg.norm(d) / numpy.linalg.norm(Y),',')
    tmp = numpy.sum(X.flatten())
    X = X/(tmp/abs(tmp))
    return X
 
def test_coil_sense_memea(input_angle):
    # b0=generate_image()[::1,::1]
    import cv2
    # b0 = numpy.array( cv2.imread('SLphantom.png',cv2.IMREAD_GRAYSCALE))[::2,::2].astype(numpy.complex64)
    
    tmp = numpy.array( cv2.imread('brain.jpg',cv2.IMREAD_GRAYSCALE))
    tmp2 = cv2.copyMakeBorder(tmp, 31,32,76,77,cv2.BORDER_CONSTANT, 0)
    b0 = tmp2[::2,::2].astype(numpy.float32)*(1.0+0.0j)

    # input_angle = -3.1

    for ii in range(0, 256):
        for jj in range(0, 256):
            if (ii - 186)**2 + (jj - 152)**2<=4**2:
                b0[ii,jj] *= numpy.exp(input_angle*1.0j)
            
    # plt.imshow(b0)
    # plt.show()
    print(b0.shape)
    # break
    #[::2,::2]
    
    S = create_fake_coils(b0.shape[0], 8)
    z = numpy.einsum('ab,abc->abc', b0, S)
#     z_prime2 = scipy.ndimage.gaussian_filter(z, (5,5,1))
    z_prime = numpy.array(scipy.ndimage.gaussian_filter(z, (22,22,1),mode='constant')) 
    # z_prime = z
    tmp = numpy.sum(z_prime.flatten())
    
    z_prime = z_prime/(tmp/abs(tmp))
    import time
    # t0 = time.time()
    #===========================================================================
    V_power = power(z_prime, maxiter=10, type='random') * 1.
    #===========================================================================
    t0 = time.time()
    V_power = power(z_prime, maxiter=10, type='random') * 1.
    t1 = time.time()
    V_lanczos91 = lanczos91(z_prime, maxiter=10, type='random', scale=1e-3) * 1.
    t2 = time.time()
    V_lanczos92 = lanczos92(z_prime, maxiter=10, type='random', scale=1e-2) * 1.
    t3=time.time()
    V_inexact = inexact_inverse_iteration(z_prime, maxiter=10, type='random') * 1.
    t4 = time.time()
    # V_eig = lanczos92(z_prime, maxiter=10, type='ones', scale=1e-4) * 1.1
    print('time=', t1-t0, t2-t1, t3-t2, t4-t3)

    tmp = numpy.sum(V_power.flatten())
    V_power = V_power/(tmp/abs(tmp))
    
    tmp = numpy.sum(V_lanczos91.flatten())
    V_lanczos91 = V_lanczos91/(tmp/abs(tmp))
    
    tmp = numpy.sum(V_lanczos92.flatten())
    V_lanczos92 = V_lanczos92/(tmp/abs(tmp))
    
    tmp = numpy.sum(V_inexact.flatten())
    V_inexact = V_inexact/(tmp/abs(tmp))
    

    x_power = solve_x(z, V_power, 50)
    x_lanczos91 = solve_x(z, V_lanczos91, 50)
    x_lanczos92 = solve_x(z, V_lanczos92, 50)
    x_inexact = solve_x(z, V_inexact, 50)
    
    print('angle of ROI:', numpy.angle(x_power[186,152]),
          numpy.angle(x_lanczos91[186,152]),
          numpy.angle(x_lanczos92[186,152]),
          numpy.angle(x_inexact[186,152]))
    
    # colorbar = numpy.einsum('a,b->ab', numpy.ones((128,1)) , numpy.linespace(-numpy.pi, numpy.pi, ))
#     print(x.shape)
    # x_eig = numpy.sum(z*V_eig.conj(), axis=2)
 
    for nn in range(0, 8):
#         matplotlib.pyplot.subplot(3,8,nn+1)
#         matplotlib.pyplot.imshow(S[:,:,nn].real, vmin=-1.5, vmax=1.5)
        matplotlib.pyplot.subplot(5,8,nn+1)
        matplotlib.pyplot.imshow(colorize(z_prime[:,:,nn]/numpy.max(abs(z_prime.flatten()))))
        matplotlib.pyplot.axis('off')
        # matplotlib.pyplot.colorbar(mappable, cax, ax)
        
        
        matplotlib.pyplot.subplot(5,8,nn+9)
        matplotlib.pyplot.imshow(colorize(V_power[:,:,nn]),)
        matplotlib.pyplot.axis('off')
#         matplotlib.pyplot.axis('off')
        
        # if nn < 1:
        #     matplotlib.pyplot.ylabel('(A)')
            
            
        matplotlib.pyplot.subplot(5,8,nn+17)
        matplotlib.pyplot.imshow(colorize(V_lanczos91[:,:,nn]),)
        matplotlib.pyplot.axis('off')
#         matplotlib.pyplot.axis('off')
        
        
        # if nn < 1:
        #     matplotlib.pyplot.ylabel('(B)')
            
        matplotlib.pyplot.subplot(5,8,nn+25)
        matplotlib.pyplot.imshow(colorize(V_lanczos92[:,:,nn]), )

        matplotlib.pyplot.axis('off')
        # if nn < 1:
        #     matplotlib.pyplot.ylabel('(C)')   
        matplotlib.pyplot.subplot(5,8,nn+33)
        matplotlib.pyplot.imshow(colorize(V_inexact[:,:,nn]), )

        matplotlib.pyplot.axis('off')
            
#     matplotlib.pyplot.colorbar()
#         matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()
 
    
    matplotlib.pyplot.subplot(2,4,1)
    matplotlib.pyplot.imshow(numpy.abs(x_power))
    matplotlib.pyplot.subplot(2,4,2)
    matplotlib.pyplot.imshow(numpy.abs(x_lanczos91))
    matplotlib.pyplot.subplot(2,4,3)
    matplotlib.pyplot.imshow(numpy.abs(x_lanczos92))
    matplotlib.pyplot.subplot(2,4,4)
    matplotlib.pyplot.imshow(numpy.abs(x_inexact))
    # matplotlib.pyplot.show()
    # for pp in (0, 2, 4, 6):
    matplotlib.pyplot.subplot(2,4,5)
    matplotlib.pyplot.imshow(colorize(x_power))
    matplotlib.pyplot.subplot(2,4,6)
    matplotlib.pyplot.imshow(colorize(x_lanczos91))
    matplotlib.pyplot.subplot(2,4,7)
    matplotlib.pyplot.imshow(colorize(x_lanczos92))
    matplotlib.pyplot.subplot(2,4,8)
    matplotlib.pyplot.imshow(colorize(x_lanczos92))
    matplotlib.pyplot.show()

    return numpy.angle(x_power[186,152]), numpy.angle(x_lanczos91[186,152]),numpy.angle(x_lanczos92[186,152]),numpy.angle(x_inexact[186,152])

def create_fake_coils_3D(N, n_coil):
    
    xx,yy,zz = numpy.meshgrid(numpy.arange(0,N),numpy.arange(0,N), numpy.arange(0,N))
    
    
#     
#     ZZ= numpy.exp(-((xx-128)/32)**2-((yy-128)/32)**2)     
    coil_sensitivity = numpy.zeros((N,N,N, n_coil)).astype(numpy.complex64)
    
    #image_sense = ()
    
    r = N/3
    phase_factor = 2
    for nn in range(0,n_coil):
        
        tmp_angle = nn*2*numpy.pi/n_coil
        shift_r = int(N/3)
        shift_x= (numpy.cos(tmp_angle)*shift_r).astype(numpy.complex64)
        shift_y= (numpy.sin(tmp_angle)*shift_r).astype(numpy.complex64)
#         ZZ= numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)
        ZZ= numpy.exp(0.2j * numpy.random.randn())* numpy.exp(-phase_factor *1.0j*((xx-N/2-shift_x)/N + (yy-N/2-shift_y)/N)  )* numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2-((zz-N/2)/r)**2).astype(numpy.complex64)  
#         coil_sensitivity +=(numpy.roll(numpy.roll(ZZ,shift_x,axis=0),shift_y,axis=1),)
        coil_sensitivity[:,:,:,nn] = ZZ
            
    return coil_sensitivity    

def test_3D_GEP():
    import numpy 
    import matplotlib.pyplot as pyplot
    from matplotlib import cm
    gray = cm.gray
    import pkg_resources
    DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
    image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0']#[0::2, 0::2, 0::2]
    image = numpy.array(image[::-1,::-1,:], order='C')
    print(image.shape)
    coil_sense = create_fake_coils_3D(128, 8)
    
    b = numpy.einsum('abcd, abc -> abcd', coil_sense, image)
    ## smooth b
    b_prime = numpy.array(scipy.ndimage.gaussian_filter(b, (10,10,10,1),mode='wrap'))
    b_prime = b_prime/numpy.max(abs(b_prime))
    
    import time
    t0 =time.time()
    # V = inexact_inverse_iteration3D(b_prime, maxiter=3)
    
    V = lanczos92(b_prime, maxiter=12, type='random', scale=1.0)
    
    print(time.time() - t0)
    image = image / numpy.max(image)
    b = b/numpy.max(abs(b))
    for nn in range(0, 8):
        pyplot.subplot(4,8,nn+1)
        pyplot.imshow(colorize(2*b[:,64,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+9)
        pyplot.imshow(colorize(2*b_prime[:,64,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+17)
        pyplot.imshow(colorize(4*V[:,64,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+25)
        pyplot.imshow(colorize(4*image[:,64,:]*V[:,64,:,nn]))
        pyplot.axis('off')
    pyplot.show()
    
    for nn in range(0, 8):
        pyplot.subplot(4,8,nn+1)
        pyplot.imshow(colorize(2*b[:,:,64,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+9)
        pyplot.imshow(colorize(2*b_prime[:,:,64,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+17)
        pyplot.imshow(colorize(4*V[:,:,64,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+25)
        pyplot.imshow(colorize(4*image[:,:,64]*V[:,:,64,nn]))
        pyplot.axis('off')
    pyplot.show()
    
    for nn in range(0, 8):
        pyplot.subplot(4,8,nn+1)
        pyplot.imshow(colorize(2*b[64,:,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+9)
        pyplot.imshow(colorize(2*b_prime[64,:,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+17)
        pyplot.imshow(colorize(4*V[64,:,:,nn]))
        pyplot.axis('off')
        pyplot.subplot(4,8,nn+25)
        pyplot.imshow(colorize(4*image[64,:,:]*V[64,:,:,nn]))
        pyplot.axis('off')
    pyplot.show()
def plot_phase_reliability():
    out_angle = numpy.load('krylov_fun_phase_reliability.npz')["out_angle"]
    input_angle = numpy.load('krylov_fun_phase_reliability.npz')["input_angle"]
    # out_angle[-2, 3] += numpy.pi
    from scipy import stats
    
    
    print(out_angle.shape)    
    print(input_angle.shape)
    import matplotlib.pyplot as plt
    # plt.show()
    # out_angle[0,0:3] -= 2*numpy.pi
    out_angle[0,-1] -= 2*numpy.pi

    
    for pp in range(0, 4):
        res1 = stats.pearsonr(input_angle, out_angle[:,pp])
        print(res1)
    plt.plot(input_angle[:], out_angle[:,0],'x')
    plt.plot(input_angle[:], out_angle[:,1],'r--')
    plt.plot(input_angle[:], out_angle[:,2],'b:')
    plt.plot(input_angle[:], out_angle[:,3],'D')
    plt.show()
    
    x = input_angle
    A = numpy.vstack([x, numpy.ones(len(x))]).T
    for pp in range(0, 4):
        y = out_angle[:,pp]
        alpha = numpy.dot((numpy.dot(numpy.linalg.inv(numpy.dot(A.T,A)),A.T)),y)
        print(alpha)
        # plt.figure(figsize = (10,8))
        plt.subplot(2,2, pp +1)
        plt.plot(x, y, 'D')
        plt.plot(x, alpha[0]*x + alpha[1], 'k')
        # plt.title('Alg. '+str(pp+1))
        plt.text(-2.5,2.5,"Alg. "+str(pp+1))
        plt.text(-0.5,2,"r={:.3f}".format(stats.pearsonr(x,y)[0]))
        plt.xlabel('$\\theta_i$ (rad)')
        
        plt.ylabel('$\\theta_e$ (rad)')
    plt.show()
    # numpy.savez('krylov_fun_phase_reliability.npz', out_angle=out_angle, input_angle=input_angle)
    
if __name__ == '__main__':
    
    test_3D_GEP()

    # test_coil_sense_memea(2.0) # angle
    
   
    
    #
    # input_angle = ()
    # out_angle = ()
    # for input in numpy.linspace(-pi, pi, 10):
    #     out_angle += (test_coil_sense_memea(input), ) # 2024 MeMeA
    #     input_angle += (input, )
    #
    # out_angle = numpy.array(out_angle)
    # input_angle = numpy.array(input_angle)
    #
    #
    # numpy.savez('krylov_fun_phase_reliability.npz', out_angle=out_angle, input_angle=input_angle)
    
    # plot_phase_reliability()
 
