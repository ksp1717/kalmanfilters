'''
_A - state transistion (Ax)
_B - input (Bu)
_H - output (Hx)
_x - state estimate (xhat)
_P - covariance (APA')
_Q - Process error covariance (BQB')
_R - Measurement error covariance (E(vv'))
'''
import random
import numpy
import pylab
import math
import scipy.linalg as sp
import numpy.random as npr
import numpy.linalg as lin
import sklear.meterics.mean_squared_error as mse


#===============================================================================================
# FUNCTIONS DEFINITION
#===============================================================================================
class Func:
  def __init__(self,_dt,_a0,_epsilon,_N,_w):
      self.dt = _dt
      self.a0 = _a0
      self.eps = _epsilon
      self.N = _N
      self.omega = _w*numpy.pi
      self.al = 0.001
      self.beta = 2
  def GetF(self): #Linearized State Matrix F
    return numpy.matrix([[1,self.dt],[(1-3*self.eps*numpy.power(self.current_state_estimate.item(0),2))*self.dt,1]])
  #def Getf(self):
    #return numpy.matrix([[mu_.item(0)],[mu_.item(1)]])
#===============================================================================================
# KALMAN FILTER
#===============================================================================================
class KF:
  def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
    self.A = _A                      # State transition matrix.
    self.B = _B                      # Control matrix.
    self.H = _H                      # Observation matrix.
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
  def KFGetCurrentState(self):
    return self.current_state_estimate
  def KFStep(self,control_vector,measurement_vector):
    #---------------------------Prediction step-----------------------------
    predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_vector
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * numpy.transpose(self.A) + self.Q
    #--------------------------Observation step-----------------------------
    innovation = measurement_vector - self.H*predicted_state_estimate
    innovation_covariance = self.H*predicted_prob_estimate*numpy.transpose(self.H) + self.R
    #-----------------------------Update step-------------------------------
    kalman_gain = predicted_prob_estimate * numpy.transpose(self.H) * numpy.linalg.inv(innovation_covariance)
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    # We need the size of the matrix so we can make an identity matrix.
    size = self.current_prob_estimate.shape[0]
    # eye(n) = nxn identity matrix.
    self.current_prob_estimate = (numpy.eye(size)-kalman_gain*self.H)*predicted_prob_estimate

#===============================================================================================
# EXTENDED KALMAN FILTER
#===============================================================================================
class EKF(Func):
  def __init__(self, _H, _x, _P, _Q, _R):
    self.H = _H                      # Observation matrix. (only for current problem)
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
  def EKFGetCurrentState(self):
    return self.current_state_estimate
  def EKFStep(self,control_vector,measurement_vector):
    #---------------------------Prediction step-----------------------------
    self.x1 = self.current_state_estimate.item(0)
    self.x2 = self.current_state_estimate.item(1)
    self.px1 = (self.x1 + self.x2 * self.dt)
    self.px2 = (self.x2 + self.dt * (self.x1 - self.eps * self.x1 * self.x1 * self.x1 + control_vector))
    predicted_state_estimate = numpy.matrix([[self.px1],[self.px2]])
    # Linearize f at xh(k-1) to get matrix A
    #self.A = self.EKFGetF()
    self.A = self.GetF()
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * numpy.transpose(self.A) + self.Q
    #--------------------------Observation step-----------------------------
    innovation = measurement_vector - self.px1
    innovation_covariance = self.H*predicted_prob_estimate*numpy.transpose(self.H) + self.R
    #-----------------------------Update step-------------------------------
    kalman_gain = predicted_prob_estimate * numpy.transpose(self.H) * numpy.linalg.inv(innovation_covariance)
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    # We need the size of the matrix so we can make an identity matrix.
    size = self.current_prob_estimate.shape[0]
    # eye(n) = nxn identity matrix.
    self.current_prob_estimate = (numpy.eye(size)-kalman_gain*self.H)*predicted_prob_estimate
  #def EKFGetF(self):
   # return numpy.matrix([[1,self.dt],[(1-3*self.eps*numpy.power(self.current_state_estimate.item(0),2))*self.dt,1]])

#===============================================================================================
# UNSCENTED KALMAN FILTER
#===============================================================================================
class UKF:
  def __init__(self,_x, _P, _Q, _R):
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
    self.L = numpy.size(_x)+numpy.size(_Q)+numpy.size(_R)
    self.l = self.al*self.al*(self.L)-self.L
    self.w = [self.l/(self.L+self.l),1/(2*(self.l+self.L))]
  def UKFGetCurrentState(self):
    return self.current_state_estimate
  def UKFStep(self,control_vector,_measurement):
    #---------------------------Augmenting step-----------------------------
    xa = numpy.mat([[self.current_state_estimate.item(0)],[self.current_state_estimate.item(1)],[0],[0]])
    temp_ = numpy.mat([[self.Q.item(0),0],[0,self.R]])
    Pa = numpy.bmat([[self.current_prob_estimate,numpy.zeros([2,2])],[numpy.zeros([2,2]),temp_]])
    #---------------------------Sigma points from previous estimate---------------------------
    deviation = sp.sqrtm((self.L+self.l)*Pa)
    sigma = xa
    sigma = numpy.append(sigma,deviation+xa,axis=1)
    sigma = numpy.append(sigma,xa-deviation,axis=1)
    #---------------------------Prediction step-----------------------------
    (sigmakx,predicted_x) = self.UKFGetxk(sigma,control_vector)
    sigmak = numpy.concatenate((sigmakx,sigma[range(self.L-2,self.L),]))
    predicted_p = self.UKFGetPk(sigmak,predicted_x)
    #-----------Sigma points prediction-------------  
    predicted_measurement = self.UKFGetyk(sigmak)
    innovation = _measurement - predicted_measurement
    #-----------------------------Update step-------------------------------
    (kalman_gain,Pyy,Pxy) = self.UKFGetK(sigmak,predicted_measurement,predicted_x)
    self.current_state_estimate = predicted_x + kalman_gain * innovation
    # We need the size of the matrix so we can make an identity matrix.
    size = self.current_prob_estimate.shape[0]
    # eye(n) = nxn identity matrix.
    self.current_prob_estimate = predicted_p - kalman_gain*Pyy*numpy.transpose(kalman_gain)
  def UKFGetxk(self,sig_,control_):
    #function to move the sigma points forward
    # and compute the predicted xk
    xhk_ = numpy.zeros([2,1])
    xik1_ = []
    xik2_ = []
    for i in range(numpy.size(sig_,axis=1)):
      x1=sig_.item(0,i)
      x2=sig_.item(1,i)
      wk=sig_.item(2,i)
      xhi1 = x1+x2*self.dt
      xhi2 = x2+self.dt*(x1-self.eps*numpy.power(x1,3)+control_+wk)
      xik1_.append(xhi1)
      xik2_.append(xhi2)
      if i == 0:
        w = self.w[0]
      else:
        w = self.w[1]
      #SigmaXi = SigmaXi + wf(xi(k-1),u(k-1))
      xhk_ = xhk_ + w*numpy.matrix([[xhi1],[xhi2]])
    return numpy.matrix([xik1_,xik2_]),xhk_

  def UKFGetPk(self,sig_,xk_):
    Pk_ = numpy.zeros([2,2])
    for i in range(numpy.size(sig_,axis=1)):
      if i == 0:
        w = self.w[0]+(1-(self.al*self.al)+self.beta)
      else:
        w = self.w[1]
      x = numpy.matrix([[sig_.item(0,i)],[sig_.item(1,i)]])
      Pk_ = Pk_ + w*(x-xk_)*numpy.transpose(x-xk_)
    return Pk_
  
  def UKFGetyk(self,sig_):
    yk_ = numpy.zeros(1)
    for i in range(numpy.size(sig_,axis=1)):
      x1=sig_.item(0,i)
      wk=sig_.item(3,i)
      if i == 0:
        w = self.w[0]
      else:
        w = self.w[1]
      yk_ = yk_ + w*(x1+wk)     #since y = x1
    return yk_
  
  def UKFGetK(self,sig_,yk_,xk_):
    pyy = numpy.zeros(1)
    pxy = numpy.zeros([2,1])
    for i in range(numpy.size(sig_,axis=1)):
      x1=sig_.item(0,i)
      x2=sig_.item(1,i)
      wk=sig_.item(3,i)
      if i == 0:
        w = self.w[0]+(1-self.al*self.al+self.beta)
      else:
        w = self.w[1]
      pyy = pyy + w*(x1+wk-yk_)*numpy.transpose(x1+wk-yk_)
      pxy = pxy + w*(numpy.matrix([[x1],[x2]])-xk_)*numpy.transpose(x1+wk-yk_)
    return pxy/pyy,pyy,pxy

#===============================================================================================
# ENSEMBLE KALMAN FILTER
#===============================================================================================
class EnUKF(Func):
  def __init__(self,_x,_P,_R):
    self.current_state_estimate = _x
    self.current_prob_estimate = _P
    self.R = _R
  def EnUKFGetCurrentState(self):
    return self.current_state_estimate
  def EnUKFStep(self,_control,_measurement):
    #Sampling
    x1 = self.current_state_estimate.item(0)
    x2 = self.current_state_estimate.item(1)
    mean = [x1,x2]
    X_ = numpy.transpose(npr.multivariate_normal(mean,self.current_prob_estimate,self.N))
    mu_ = numpy.transpose(numpy.matrix((X_.sum(axis=1)/self.N)))
    #Prediction
    (xik,predicted_x) = self.EnUKFGetxk(X_,_control)
    predicted_p = ((xik-predicted_x)*numpy.transpose(xik-predicted_x))/(self.N)
    #Covariances Pxy and Pyy
    yi = numpy.matrix(xik[0,:])    #Predicted Update (x1 for this problem)
    yh = numpy.sum(yi)/self.N
    pyx = ((yi-yh)*numpy.transpose(xik-predicted_x))/(self.N-1)
    pyy = (yi-yh)*numpy.transpose(yi-yh)/(self.N-1)
    #Update
    self.current_state_estimate = predicted_x + numpy.transpose(pyx)*(_measurement-yh)/(pyy+self.R)
    self.current_prob_estimate = predicted_p - (numpy.transpose(pyx)/(pyy+self.R))*pyx
  def EnUKFGetxk(self,X,control_):
    #xhatk-
    xhkm_ = numpy.zeros([2,1])
    #xi(k-1) propagated to xi(k)-
    xik1_ = []
    xik2_ = []
    for i in range(numpy.size(X,axis=1)):
      x1=X.item(0,i)
      x2=X.item(1,i)
      xhi1 = x1+x2*self.dt
      xhi2 = x2+self.dt*(x1-self.eps*numpy.power(x1,3)+control_)
      xik1_.append(xhi1)
      xik2_.append(xhi2)
      xhkm_ = xhkm_ + numpy.matrix([[xhi1],[xhi2]])
    #find the mean=xk-
    xhkm_ = xhkm_/self.N
    return numpy.matrix([xik1_,xik2_]),xhkm_

#===============================================================================================
# PARTICLE ENSEMBLE KALMAN FILTER
#===============================================================================================  
class EnKF(Func):
  def __init__(self,_x,_P,_Q,_R,_H,_N):
    self.current_state_estimate = _x
    self.current_prob_estimate = _P
    self.N = _N
    self.Q = _Q
    self.R = _R
    self.H = _H    #Used for both NLmeasurement and estimated y
  def EnKFGetCurrentState(self):
    return self.current_state_estimate
  def EnKFStep(self,_control,_measurement):
    #SamplingEnsemble------------------------
    (X,Xmu,Pxx) = self.EnKFGetEn()
    #Innovation-------------------------------
    innovation = (npr.normal(_measurement,self.R,self.N)-self.H*X)
    #Update the Ensemble---------------------
    kalman_gain = Pxx*numpy.transpose(self.H)*lin.inv(self.H*Pxx*numpy.transpose(self.H)+self.R)
    Xp = X + kalman_gain*innovation
    self.current_prob_estimate = Pxx + kalman_gain*self.H*Pxx
    self.current_state_estimate = Xp.sum(axis=1)/self.N
  def EnKFGetEn(self):
    x1 = self.current_state_estimate.item(0)
    x2 = self.current_state_estimate.item(1)
    mean = [x1,x2]
    X_ = numpy.transpose(npr.multivariate_normal(mean,self.current_prob_estimate,self.N))
    mu_ = numpy.transpose(numpy.matrix((X_.sum(axis=1)/self.N)))
    A_ = (X_-mu_)
    C_ = A_*numpy.transpose(A_)/(self.N-1)
    return X_,mu_,C_
