#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:17:39 2022

@author: rafiul
"""


# import scipy.integrate as integrate
# from scipy.integrate import odeint

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
#import matplotlib
#from numba import jit
from scipy import special 
from scipy.interpolate import interp1d

# f = interp1d(nth_car_data['time'],nth_car_data['speed'])


######################################

# Functions  

# no need to modify--Runge Kutta method 
# @jit(nopython=True)
# def RK4(func, X0, ts):
#        """
#        Runge Kutta 4 solver.
#        """
     
#        dt = ts[1] - ts[0]
#        nt = len(ts)
#        X  = np.zeros((nt, X0.shape[0]),dtype=np.float64)
#        X[0] = X0
#        for i in range(nt-1):
#            k1 = func(X[i], ts[i])
#            k2 = func(X[i] + dt/2. * k1, ts[i] + dt/2.)
#            k3 = func(X[i] + dt/2. * k2, ts[i] + dt/2.)
#            k4 = func(X[i] + dt    * k3, ts[i] + dt)
#            X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
#        return X 


#  see this link for model and paramterts https://en.wikipedia.org/wiki/Intelligent_driver_model  
# DOI: 10.1098/rsta.2010.0084 



# @jit(nopython=True)
def idm_model(x,t):
    X,V = x[0],x[1]
    dX,dV = np.zeros(1,dtype=np.float64), np.zeros(1,dtype=np.float64)
    
    
    dX = V # Differtial Equation 1
    ###
    s = position_LV(t) - X - 5 # 5 = length of the car
    deltaV = V - speed_LV(t)
    sstar = s0+V*T + (V*deltaV)/(2*np.sqrt(a*b))
    # ###
    dV = a*(1-(V/V_0)**delta - (sstar/s)**2) # Differtial Equation 2
    
    return np.array([dX,dV],dtype=np.float64) 


# @jit(nopython=True)

def speed_LV(t):
    return interp1d(nth_car_data['time'],nth_car_data['speed'],bounds_error=False)(t) 

def position_LV(t):
    return interp1d(nth_car_data['time'],postion_of_the_LV,bounds_error=False)(t)  


def fractional_idm_model_1d(V,t,X):    
    # index = round(t) #convert into integer number 
    
    current_position_of_follower = X 
    ###
    s = position_LV(t) - current_position_of_follower - 5 # 5 = length of the car
    deltaV = V - speed_LV(t)
    sstar = s0+V*T + (V*deltaV)/(2*np.sqrt(a*b))
    
    # ###
    dV = a_alpha*(1-(V/V_0)**delta - (sstar/s)**2) # Differtial Equation 2
    
    return dV

def caputoEuler_1d(a, f, y0, tspan, x0):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
    D^a y(t) = f(y,t)
    Args:
      a: fractional exponent in the range (0,1)
      f: callable(y,t) returning a numpy array of shape (d,)
         Vector-valued function to define the right hand side of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.
    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    Raises:
      FODEValueError
    See also:
      K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
         method
      C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
         Differential Equations
    """
    #(d, a, f, y0, tspan) = _check_args(a, f, y0, tspan)
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    c = special.rgamma(a) * np.power(h, a) / a
    w = c * np.diff(np.power(np.arange(N), a))
    fhistory = np.zeros(N - 1, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    y[0] = y0;
    x[0] = x0;
    for n in range(0, N - 1):
        tn = tspan[n]
        yn = y[n]
        fhistory[n] = f(yn, tn, x[n])
        y[n+1] = y0 + np.dot(w[0:n+1], fhistory[n::-1])
        x[n+1] = x[n] + y[n+1] * h
    return np.array([x,y])



######################################



# Global  variables 

#  see this link for model and paramterts https://en.wikipedia.org/wiki/Intelligent_driver_model 

a=1.5
a_alpha = 1.5
V_0 = 20 # desired speed m/s
delta =  4.0
b = 1.67 
T = 1.5 
s0=30

# ######################################

# Actual data 
import pandas as pd 

df = pd.read_csv('RAllCarDataTime350.csv') 
df.head()

nth_car = 2

nth_car_data = df.loc[df['nthcar'] == nth_car, :]  

nth_car_speed = np.array(df.loc[df['nthcar'] == nth_car,'speed'])  



# leader vehicle profile 
# 7 m/s - 25.2 km/h  11 m/s - 39.6 km/h  18 m/s - 64.8 km/h 22 m/s - 79.2 km/h 
# 25 km/h -- 6.95 m/s 40 km/h -- 11.11 m/s 60 km/h -- 16.67 m/s 

# dt=1 #time step -- 1 sec 


time_span = np.array(nth_car_data['time'])
dt = time_span[1]-time_span[0]

# speed_of_the_LV = 15*np.ones(600+1) # we will need data

# speed_of_the_LV = np.concatenate((np.linspace(0,7,60),7*np.ones(120),np.linspace(7,11,60), 11*np.ones(120), np.linspace(11,0,60) ))# we will need data

speed_of_the_LV = nth_car_speed

num_points = len(speed_of_the_LV) 

postion_of_the_LV = np.zeros(num_points) 


initla_position_of_the_LV = 100.0

postion_of_the_LV[0] = initla_position_of_the_LV

for i in range(1,num_points):
      
      postion_of_the_LV[i] = postion_of_the_LV[i-1] + dt*(speed_of_the_LV[i]+speed_of_the_LV[i-1])/2 
 
plt.figure() 
plt.subplot(211) 
plt.plot(speed_of_the_LV)
plt.xlabel('time')
plt.ylabel('speed of the leader vehicle') 

plt.subplot(212) 
plt.plot(postion_of_the_LV)
plt.xlabel('time')
plt.ylabel('postion of the leader vehicle') 

    


#### 


# simulation_time = 35

# time_span = np.linspace(0, simulation_time, int(simulation_time /dt)+1) 



initial_position = 0.
initial_velocity = 0.
x0 = np.array([initial_position,initial_velocity],dtype=np.float64) #initial position and velocity


f,ax=plt.subplots(2,1,figsize=(10,10))

ax[0].plot(time_span,postion_of_the_LV,label = 'position of the LV') 
ax[1].plot(time_span,speed_of_the_LV,label = 'speed of the LV') 
# Classical ODE
sol = integrate.odeint(idm_model, x0, time_span) 

ax[0].plot(time_span,sol[:,0], label='position of the FV using alpha = 1.00') 
ax[1].plot(time_span,sol[:,1], label='speed of the FV using alpha = 1.00') 


x0 = np.array([initial_velocity],dtype=np.float64) #initial position and velocity

# Fractional ODE
for alpha in [.95,.9,.8]:
    #sol = fintegrate_mod.fodeint_mod(alpha,fractional_idm_model_mod, x0, ts) #, args=(number_groups,beta_P,beta_C,beta_A,v,w,mu_E,mu_A,mu_P,mu_C,p,q,contact_by_group))
    sol = caputoEuler_1d(alpha,fractional_idm_model_1d, initial_velocity, time_span, initial_position) #, args=(number_groups,beta_P,beta_C,beta_A,v,w,mu_E,mu_A,mu_P,mu_C,p,q,contact_by_group))
    ax[0].plot(time_span,sol[0], label='position of the FV using alpha = %.2f' %alpha) 
    ax[1].plot(time_span,sol[1], label='speed of the FV using alpha = %.2f' %alpha) 

ax[0].set_xlabel('time (sec)')
ax[0].set_ylabel('posiition (m)')
ax[0].legend()

ax[1].set_xlabel('time (sec)')
ax[1].set_ylabel('speed (m/s)')
ax[1].legend() 

# plt.savefig('simulation.pdf',dpi=300) 






