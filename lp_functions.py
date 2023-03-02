'''
Copyright 2023 Mohak Goyal
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np

def psi_func_lmsr(px, py):  
    # This function return the belief function for which the LMSR based CFMM is optimal
    # It also divides the output by py as per the expression for expected CFMM inefficiency
    psi = (px*py)/np.power(px+py,2)
    return psi/py

def psi_func_constant_product(px, py):
    # This function return the constant belief function for which the constant product CFMM is optimal
    # The function also divides the output by py as per the expression for expected CFMM inefficiency
    psi =  (np.power(px/py, 0))
    return psi/py

def psi_func_constant_weighted_product(px, py, alpha = 2):
    # This function return the belief function for which the constant weighted product CFMM is optimal
    # alpha is the weight of x, as in x^{\alpha} y = K is the CFMM trading function
    # The function also divides the output by py as per the expression for expected CFMM inefficiency
    psi =  np.power(px/py, (alpha-1)/(alpha+1)) 
    return psi/py

def psi_func_blackScholes(px, py, sigma_x=1, sigma_y=1, mu_x=0, mu_y=0,gamma=1, T_stop = 10):
    # This function calculates and returns the belief function complied from time-discounting of the 
    # Black Scholes price dynamics (independent processes for px and py)
    # sigma_x --> standard deviation of the Brownian motion of log(pX)
    # sigma_y --> standard deviation of the Brownian motion of log(pY)
    # mu_x --> mean of the Brownian motion of log(pX)
    # mu_y --> mean of the Brownian motion of log(pX)
    # gamma --> time discounting factor
    # T_stop --> The time upto which the CFMM is run
    # The function also divides the output by py as per the expression for expected CFMM inefficiency
    grid_len = 200
    T = np.linspace(0,T_stop,grid_len)
    psi = np.zeros_like(px/py)
    dt = T_stop/grid_len

    for i in range(len(T)-1):
      t = (T[i] + T[i+1])/2
      psi += dt*np.exp(-gamma*t-np.power((np.log(px)-(mu_x-0.5*sigma_x**2)*t),2)/(2*sigma_x**2 *t) \
                               -np.power((np.log(py)-(mu_y-0.5*sigma_y**2)*t),2)/(2*sigma_y**2 *t)) \
               *(1/(2*np.pi))*(1/(px*py*sigma_x*sigma_y*t))
    return psi/py

def psi_func_concentrated(px, py, pmax=2, pmin=0.5):
    # The concentrated-liquidity belief
    # pmax --> The upper range of exchange rate in the LP's position
    # pmin --> The lower range of exchange rate in the LP's position
    # The function also divides the output by py as per the expression for expected CFMM inefficiency
    psi =  (np.power(px/py, 0)) 
    psi = np.where(px/py > pmax , 0, psi)
    psi = np.where(px/py < pmin , 0, psi)

    return psi/py

def exchange_rate(px, py):
    # returns p = px/py
    return px/py

def area(dpx,dpy):
    # used for finding the area of a cell on the grid
    return dpx*dpy

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()
