import numpy as np
from numpy import exp
from math import pi, log, sqrt
from scipy import integrate
from scipy.optimize import fsolve


def g_i(freq, tau, epsilon, rbf_type):
    
    alpha = 2*pi*freq*tau  
    
    rbf_switch = {
                'gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0_matern': lambda x: exp(-abs(epsilon*x)),
                'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                }
    
    rbf = rbf_switch.get(rbf_type)
    
    # check 
    # seems factor 0.5 was missing
    integrand_g_i = lambda x: 1./(1.+(alpha**2)*exp(2.*x))*rbf(x)
    
    out_val = integrate.quad(integrand_g_i, -50, 50, epsabs=1E-6, epsrel=1E-6)
    
    return out_val[0]

def g_ii(freq, tau, epsilon, rbf_type):
    
    alpha = 2*pi*freq*tau  
    
    rbf_switch = {
                'gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0_matern': lambda x: exp(-abs(epsilon*x)),
                'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                }
    
    rbf = rbf_switch.get(rbf_type)

    # check 
    # seems factor 0.5 was missing
    integrand_g_ii = lambda x: alpha/(1./exp(x)+(alpha**2)*exp(x))*rbf(x)
    
    out_val = integrate.quad(integrand_g_ii, -50, 50, epsabs=1E-6, epsrel=1E-6)
    
    return out_val[0]

def compute_epsilon(tau_vec, rbf_type):
    
    N_taus = tau_vec.shape[0]
    rbf_switch = {
                'gaussian': lambda x: exp(-(x)**2)-0.5,
                'C0_matern': lambda x: exp(-abs(x))-0.5,
                'C2_matern': lambda x: exp(-abs(x))*(1+abs(x))-0.5,
                'C4_matern': lambda x: 1/3*exp(-abs(x))*(3+3*abs(x)+abs(x)**2)-0.5,
                'C6_matern': lambda x: 1/15*exp(-abs(x))*(15+15*abs(x)+6*abs(x)**2+abs(x)**3)-0.5,
                'inverse_quadratic': lambda x: 1/(1+(x)**2)-0.5
                }
        
    rbf = rbf_switch.get(rbf_type)
    FWHM_coeff = 2*fsolve(rbf,1)
    delta = np.mean(np.diff(np.log(tau_vec.reshape(N_taus))))
    epsilon = 0.5*FWHM_coeff/delta
    
    return epsilon
    
def inner_prod_rbf_1(freq, tau, epsilon, rbf_type): # first order derivative
    
    a = epsilon*log(freq*tau)

    rbf_switch = {
                'gaussian': -epsilon*(-1+a**2)*exp(-(a**2/2))*sqrt(pi/2),
                'C0_matern': epsilon*(1-abs(a))*exp(-abs(a)),
                'C2_matern': epsilon/6*(3+3*abs(a)-abs(a)**3)*exp(-abs(a)),
                'C4_matern': epsilon/30*(105+105*abs(a)+30*abs(a)**2-5*abs(a)**3-5*abs(a)**4-abs(a)**5)*exp(-abs(a)),
                'C6_matern': epsilon/140*(10395 +10395*abs(a)+3780*abs(a)**2+315*abs(a)**3-210*abs(a)**4-84*abs(a)**5-14*abs(a)**6-abs(a)**7)*exp(-abs(a)),
                'inverse_quadratic': 4*epsilon*(4-3*a**2)*pi/((4+a**2)**3)
                }
    
    return rbf_switch.get(rbf_type)

def inner_prod_rbf_2(freq, tau, epsilon, rbf_type): # second order derivative
    
    a = epsilon*log(freq*tau)

    rbf_switch = {
                'gaussian': epsilon**3*(3-6*a**2+a**4)*exp(-(a**2/2))*sqrt(pi/2),
                'C0_matern': epsilon**3*(1+abs(a))*exp(-abs(a)),
                'C2_matern': epsilon**3/6*(3 +3*abs(a)-6*abs(a)**2+abs(a)**3)*exp(-abs(a)),
                'C4_matern': epsilon**3/30*(45 +45*abs(a)-15*abs(a)**3-5*abs(a)**4+abs(a)**5)*exp(-abs(a)),
                'C6_matern': epsilon**3/140*(2835 +2835*abs(a)+630*abs(a)**2-315*abs(a)**3-210*abs(a)**4-42*abs(a)**5+abs(a)**7)*exp(-abs(a)),
                'inverse_quadratic': 48*(16 +5*a**2*(-8 + a**2))*pi*epsilon**3/((4 + a**2)**5)
                }
    
    return rbf_switch.get(rbf_type)

def gamma_to_x(gamma_vec, tau_vec, epsilon, rbf_type): 
    
    if rbf_type == 'pwl':
        x_vec = gamma_vec
        
    else:
        rbf_switch = {
                    'gaussian': lambda x: exp(-(epsilon*x)**2),
                    'C0_matern': lambda x: exp(-abs(epsilon*x)),
                    'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                    'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                    'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        B = np.zeros([N_taus, N_taus])
        
        for p in range(0, N_taus):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)
                
        B = 0.5*(B+B.T)
                
        x_vec = np.linalg.solve(B, gamma_vec)
            
    return x_vec

def x_to_gamma(x_vec, tau_vec, epsilon, rbf_type): 
    
    if rbf_type == 'pwl':
        gamma_vec = x_vec
        
    else:
        rbf_switch = {
                    'gaussian': lambda x: exp(-(epsilon*x)**2),
                    'C0_matern': lambda x: exp(-abs(epsilon*x)),
                    'C2_matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                    'C4_matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                    'C6_matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'inverse_quadratic': lambda x: 1/(1+(epsilon*x)**2)
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size    
        B = np.zeros([N_taus, N_taus])
        
        for p in range(0, N_taus):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)
                
        B = 0.5*(B+B.T)
                
        gamma_vec = np.matmul(B, x_vec)
            
    return gamma_vec

def A_re(freq_vec, tau_vec, epsilon, discr_method):
    
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    out_A_re = np.zeros((N_freqs, N_taus))
    
    for p in range(0, N_freqs):
        for q in range(0, N_taus):

            if discr_method == 'pwl':
                if q == 0:
                    out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
            else:
                out_A_re[p, q] = g_i(freq_vec[p], tau_vec[q], epsilon, discr_method)
                
    return out_A_re

def A_im(freq_vec, tau_vec, epsilon, discr_method):
    
    omega_vec = 2.*pi*freq_vec

    N_taus = tau_vec.size
    N_freqs = freq_vec.size

    out_A_im = np.zeros((N_freqs, N_taus))
    
    for p in range(0, N_freqs):
        for q in range(0, N_taus):
            if discr_method == 'pwl':
                if q == 0:
                    out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
            else:
                out_A_im[p, q] = - g_ii(freq_vec[p], tau_vec[q], epsilon, discr_method)
    
    return out_A_im

def L(tau_vec):
    
    N_taus = tau_vec.size
    out_L = np.zeros((N_taus-2, N_taus))
    
    for p in range(0, N_taus-2):

        delta_loc = log(tau_vec[p+1]/tau_vec[p])
        
        if p==0 or p == N_taus-3:
            out_L[p,p] = 2./(delta_loc**2)
            out_L[p,p+1] = -4./(delta_loc**2)
            out_L[p,p+2] = 2./(delta_loc**2)
        else:
            out_L[p,p] = 1./(delta_loc**2)
            out_L[p,p+1] = -2./(delta_loc**2)
            out_L[p,p+2] = 1./(delta_loc**2)

    return out_L

def assemble_M(freq_vec, tau_vec, epsilon, rbf_type):
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    out_M = np.zeros([N_freqs, N_taus])
    for n in range(0, N_freqs):
        for m in range(0, N_taus):            
            out_M[n,m] = inner_prod_rbf_2(freq_vec[n], freq_vec[m], epsilon, rbf_type)
    return out_M

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)