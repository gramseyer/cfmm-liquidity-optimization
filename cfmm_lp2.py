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


This script requires the cvxpy library which can be installed via
pip install cvxpy

We have implemented the Black Scholes based belief, constant product, concentrated liquidity, LMSR and 
Cosntant weighted product based belief
Custom belief functions can be defined in the function psi_func_custom below
This script plots optimal liquidity allocation and the correspinding CFMM trading functions for four examples 
'''
import cvxpy as cp
import numpy as np
from lp_functions import find_nearest, area, exchange_rate, psi_func_concentrated, psi_func_lmsr, psi_func_constant_product, psi_func_constant_weighted_product, psi_func_blackScholes 
import matplotlib.pyplot as plt


'''
 psi is the belief function defined on p_x, p_Y \in (0,1]x(0,1]

'''

def psi_func_custom(px, py):
    ####### A user can give a belief function here
    # In the default example, we have the concentrated-liquidity belief
    # The function also divides the output by py as per the expression for expected CFMM inefficiency
    psi =  (np.power(px/py, 0)) 
    psi = np.where(px/py > 2 , 0, psi)
    psi = np.where(px/py < 0.5 , 0, psi)

    return psi/py

n = 400     # grid points
p = np.logspace(-4,4, 2*n+1)
p_diff = np.asarray([(p[i+1] - p[i]) for i in range(2*n)])
p_center = np.asarray([(p[i+1] + p[i])/2 for i in range(2*n)])

# Wlog (since the assets are divisible) we assume the initial exchange rates are P_X =P_Y =1

L = cp.Variable(2*n)        # liquidity
X0 = cp.Variable(1)         # Initial amount of X in CFMM
Y0 = cp.Variable(1)         # Initial amount of Y in CFMM
B = 2

p1_diff = p_diff[:n]
p2_diff = p_diff[n:]

p1_center = p_center[:n]
p2_center = p_center[n:]

px = np.logspace(-4, 0, 2*n+1)
px_diff = np.asarray([(px[i+1] - px[i]) for i in range(2*n)])
px_center = np.asarray([(px[i+1] + px[i])/2 for i in range(2*n)])

py = np.logspace(-4, 0, 2*n+1)
py_diff = np.asarray([(py[i+1] - py[i]) for i in range(2*n)])
py_center = np.asarray([(py[i+1] + py[i])/2 for i in range(2*n)])

ex_rate = exchange_rate(px_center[:,None], py_center[None,:])
area_of_cell = area(px_diff[:,None], py_diff[None,:])



## initializing belief functions

psi_lmsr = psi_func_lmsr(px_center[:,None], py_center[None,:])
psi_cwp = psi_func_constant_weighted_product(px_center[:,None], py_center[None,:], alpha = 2)
psi_cp = psi_func_constant_product(px_center[:,None], py_center[None,:])
psi_custom = psi_func_custom(px_center[:,None], py_center[None,:])


## Converting the belief function from a 2D grid on px,py to a 1D grid on p

priorP_lmsr =np.zeros_like(p_center)
priorP_cwp =np.zeros_like(p_center)
priorP_cp =np.zeros_like(p_center)
priorP_custom =np.zeros_like(p_center)

for x in range(ex_rate.shape[0]):
    for y in range(ex_rate.shape[1]):
        p_idx = find_nearest(p_center, ex_rate[x,y])
        priorP_lmsr[p_idx] += psi_lmsr[x,y]*area_of_cell[x,y]
        priorP_cwp[p_idx] += psi_cwp[x,y]*area_of_cell[x,y]
        priorP_cp[p_idx] += psi_cp[x,y]*area_of_cell[x,y]
        priorP_custom[p_idx] += psi_custom[x,y]*area_of_cell[x,y]


# Defining the objective function for minimizing the expected CFMM inefficiency for some examples of beliefs
objective_lmsr = cp.Minimize(cp.sum(cp.multiply(priorP_lmsr, cp.inv_pos(L)))) 
objective_cwp = cp.Minimize(cp.sum(cp.multiply(priorP_cwp, cp.inv_pos(L)))) 
objective_cp = cp.Minimize(cp.sum(cp.multiply(priorP_cp, cp.inv_pos(L)))) 
objective_custom = cp.Minimize(cp.sum(cp.multiply(priorP_custom, cp.inv_pos(L)))) 

constraints = [ cp.sum(cp.multiply(L[n:],p2_diff)/(p2_center*p2_center)) <= X0, 
                cp.sum(cp.multiply(L[:n],p1_diff)/p1_center) <= Y0, 
                X0+Y0 <= B, 
                L>=0
              ] 

# LMSR
prob_lmsr = cp.Problem(objective_lmsr, constraints)
prob_lmsr.solve()
L_lmsr = L.value
X0_lmsr = X0.value
Y0_lmsr = Y0.value

# Constant weighted product
prob_cwp = cp.Problem(objective_cwp, constraints)
prob_cwp.solve()
L_cwp = L.value
X0_cwp = X0.value
Y0_cwp = Y0.value

# Constant product
prob_cp = cp.Problem(objective_cp, constraints)
prob_cp.solve()
L_cp = L.value
X0_cp = X0.value
Y0_cp = Y0.value

# User specified belief function
prob_custom = cp.Problem(objective_custom, constraints)
prob_custom.solve()
L_custom = L.value
X0_custom = X0.value
Y0_custom = Y0.value

## Obtaining X(p) and Y(p) from L(p)
X_lmsr = np.zeros_like(p)
Y_lmsr = np.zeros_like(p)
X_lmsr[n]= X0_lmsr
Y_lmsr[n]= Y0_lmsr

X_cwp = np.zeros_like(p)
Y_cwp = np.zeros_like(p)
X_cwp[n]= X0_cwp
Y_cwp[n]= Y0_cwp

X_cp = np.zeros_like(p)
Y_cp = np.zeros_like(p)
X_cp[n]= X0_cp
Y_cp[n]= Y0_cp

X_custom = np.zeros_like(p)
Y_custom = np.zeros_like(p)
X_custom[n]= X0_custom
Y_custom[n]= Y0_custom

dY_lmsr = L_lmsr*p_diff/p_center
dX_lmsr = -L_lmsr*p_diff/(p_center*p_center)

dY_cwp = L_cwp*p_diff/p_center
dX_cwp = -L_cwp*p_diff/(p_center*p_center)

dY_cp = L_cp*p_diff/p_center
dX_cp = -L_cp*p_diff/(p_center*p_center)

dY_custom = L_custom*p_diff/p_center
dX_custom = -L_custom*p_diff/(p_center*p_center)

for i in range(n+1, 2*n+1):
  X_lmsr[i] = X_lmsr[i-1] + dX_lmsr[i-1]
  Y_lmsr[i] = Y_lmsr[i-1] + dY_lmsr[i-1]

  X_cwp[i] = X_cwp[i-1] + dX_cwp[i-1]
  Y_cwp[i] = Y_cwp[i-1] + dY_cwp[i-1]

  X_cp[i] = X_cp[i-1] + dX_cp[i-1]
  Y_cp[i] = Y_cp[i-1] + dY_cp[i-1]

  X_custom[i] = X_custom[i-1] + dX_custom[i-1]
  Y_custom[i] = Y_custom[i-1] + dY_custom[i-1]


for i in range(n-1, -1, -1):
  X_lmsr[i] = X_lmsr[i+1] - dX_lmsr[i]
  Y_lmsr[i] = Y_lmsr[i+1] - dY_lmsr[i]

  X_cwp[i] = X_cwp[i+1] - dX_cwp[i]
  Y_cwp[i] = Y_cwp[i+1] - dY_cwp[i]

  X_cp[i] = X_cp[i+1] - dX_cp[i]
  Y_cp[i] = Y_cp[i+1] - dY_cp[i]

  X_custom[i] = X_custom[i+1] - dX_custom[i]
  Y_custom[i] = Y_custom[i+1] - dY_custom[i]


## Plotting, for clarity and to minimize the effect of numerical errors on the plots,
## we plot only the range where the exchange rate is between 1/25 and 25

idx_l = find_nearest(p_center, 0.04)
idx_u = find_nearest(p_center, 25)


plt.plot(np.log(p_center[idx_l:idx_u]), L_lmsr[idx_l:idx_u], label = 'LMSR')
plt.plot(np.log(p_center[idx_l:idx_u]), L_cwp[idx_l:idx_u], label= 'Constant weighted product' )
plt.plot(np.log(p_center[idx_l:idx_u]), L_cp[idx_l:idx_u], label ='Constant product')
plt.plot(np.log(p_center[idx_l:idx_u]), L_custom[idx_l:idx_u], label = 'Custom CFMM' )

plt.ylabel('L(p)')
plt.xlabel('log(p)')
plt.title('L(p) on log plot')
plt.legend()
plt.grid(True)
plt.show()

plt.loglog((p_center[idx_l:idx_u]), L_lmsr[idx_l:idx_u],  label = 'LMSR')
plt.loglog((p_center[idx_l:idx_u]), L_cwp[idx_l:idx_u], label ='Constant weighted product')
plt.loglog((p_center[idx_l:idx_u]), L_cp[idx_l:idx_u], label ='Constant product')
plt.loglog((p_center[idx_l:idx_u]), L_custom[idx_l:idx_u], label = 'Custom CFMM')
plt.ylabel('log(L(p))')
plt.xlabel('log(p)')
plt.title('L(p) on log-log plot')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(X_lmsr[idx_l:idx_u], Y_lmsr[idx_l:idx_u],  label = 'LMSR') 
plt.plot(X_cwp[idx_l:idx_u], Y_cwp[idx_l:idx_u], label ='Constant weighted product') 
plt.plot(X_cp[idx_l:idx_u], Y_cp[idx_l:idx_u], label ='Constant product') 
plt.plot(X_custom[idx_l:idx_u], Y_custom[idx_l:idx_u], label = 'Custom CFMM') 
plt.ylabel('Y')
plt.xlabel('X')
plt.title('CFMM trading function')
plt.legend()
plt.grid(True)
plt.show()
