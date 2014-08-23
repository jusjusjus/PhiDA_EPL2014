#! /usr/bin/env python

import os
import numpy as np
import ctypes as ct
import scipy.optimize as opt
from scipy.weave import inline
import struct



def architecture():
	return struct.calcsize("P") * 8



def extra_arch_arg():
	if architecture() == 32:
		return ['-m32']

	else:
		return []



c_bridge_variance = """
	unsigned i, j, box_size_1, N;
	double *data_i, i1, slope, x;

	box_size_1 = box_size-1;

	N = 1+(data_size-box_size)/stride;

	for(i=0; i<N; i++)
	{
		data_i = data + i*stride;
                //printf("%i %lf ", i, *data_i);
		slope = (data_i[box_size_1]-data_i[0])/(double)box_size_1;
		i1 = (double)(i+1);

		for(j=1; j<box_size_1; j++)
		{
			x = data_i[j] - data_i[0] - (double)j*slope;
			y_out[j] = ((double)i*y_out[j] + x*x)/i1;
		}	
	}

	// last one
	
	if(stride*N+box_size-1 != data_size)
	{
		data_i = data + (data_size-box_size);
		slope = (data_i[box_size_1]-data_i[0])/(double)box_size_1;
		i1 = (double)(i+1);
	
		for(j=1; j<box_size_1; j++)
		{
			x = data_i[j] - data_i[0] - (double)j*slope;
			y_out[j] = ((double)i*y_out[j] + x*x)/i1;
		}	
	}

"""


headers = ['<math.h>', '<stdio.h>']

bv_args = ['box_size', 'data', 'data_size', 'stride', 'y_out']
def bridge_variance(data, box_size, stride=1):
	data = np.asarray(data, dtype=float)
        data_size = data.size
	box_size = int(box_size)
	stride = int(stride)
        y_out = np.zeros((box_size), float);

	inline(code=c_bridge_variance,
			arg_names=bv_args,
			headers=headers,
			extra_compile_args=extra_arch_arg(), extra_link_args=extra_arch_arg(), verbose=0)

	return np.array(y_out, dtype=float)


c_diffusion_sum="""
	unsigned i;
	double y_sum=y[0], output=0., y_sum_inv;
	
	for(i=1; i<size; i++)	
	{
		y_sum = ((double)i*y_sum + y[i])/(double)(i+1);
	}
	y_sum_inv = 1./y_sum;
	for(i=0; i<size; i++)	
	{
		output = ((double)i*output + (y_sum_inv*y[i])*(y[i]/f[i]))/(double)(i+1);
	}
	return_val = output;

"""

ds_args = ['y', 'f', 'size']
def diffusion_sum(y, f):
        size = y.size
	output = inline(code=c_diffusion_sum,
			arg_names=ds_args,
			headers=headers,
			extra_compile_args=extra_arch_arg(), extra_link_args=extra_arch_arg(), verbose=0)
	
	return output




colors = ['all', 'white', 'colored', 'pink']

bounds = {}
LARGE = 100000.
SMALL = 1./LARGE


bounds['white'] = dict()
def white_bridge_process(t, T, D=1.):
	return D*t*(1.-t/T)


def int_tt(t, tau):
	return 2.*tau*( t - tau * (1.-np.exp(-t/tau)) )


bounds['colored'] = dict(tau=(SMALL, LARGE))
def colored_bridge_process(t, T, D=1., tau=1.):
	return D*(2.*tau)**-1*((1.-2.*t/T)*int_tt(t, tau) - 2.*t/T*(tau**2*(1.-np.exp(-t/tau))*(1.-np.exp(-(T-t)/tau))) + t**2/T**2*int_tt(T, tau))


bounds['alt_colored'] = dict(tau=(SMALL, LARGE))
def alt_colored_bridge_process(t, T, D=1., tau=1.):
	return D*((int_tt(T, tau)/T)**-1*((1.-2.*t/T)*int_tt(t, tau) - 2.*t/T*(tau**2*(1.-np.exp(-t/tau))*(1.-np.exp(-(T-t)/tau)))) + t**2/T)



bounds['pink'] = dict(c=(SMALL, LARGE), alpha=(SMALL, LARGE))
def pink_bridge_process(x, X, D=1., c=1., alpha=0.7): # Z(x,x)-2x/X Z(x,X)+x^2/X^2 Z(X,X)
	alpha1, alpha2, cx, cX = 1.-alpha, 2.-alpha, c+x, c+X
	c_alpha1, cx_alpha1, X_alpha1, cX_alpha1 = c**(1.-alpha), cx**(1.-alpha), X**(1.-alpha), (c+X)**(1.-alpha)
	c_alpha2, cx_alpha2, X_alpha2, cX_alpha2 = c*c_alpha1, cx*cx_alpha1, X*X_alpha1, cX*cX_alpha1

	N = 0.5*alpha1/(cX_alpha1-c_alpha1)
	Zxx = 2.*cx/alpha1*(cx_alpha1-c_alpha1) - 2./alpha2*(cx_alpha2-c_alpha2)
	ZxX = Zxx + (c_alpha2+cX_alpha2-cx_alpha2-(c+X-x)**alpha2) / (alpha1*alpha2)
	ZXX = 2.*cX/alpha1*(cX_alpha1-c_alpha1) - 2./alpha2*(cX_alpha2-c_alpha2)

	return D*N*(Zxx - 2.*x/X*ZxX + x**2/X**2*ZXX)

###

class bridge_process(object):

	def __init__(self, color):

		color2params = dict(all=(None, dict()),\
			  	white=(white_bridge_process, dict(D=1.)),\
				colored=(colored_bridge_process, dict(D=1., tau=1.)),\
				alt_colored=(alt_colored_bridge_process, dict(D=1., tau=1.)),\
			   	#pink=(pink_bridge_process, dict(D=1., c=1., alpha=0.7)))
			   	pink=(pink_bridge_process, dict(D=1., alpha=0.7)))

		self.bridge, self.params = color2params[color]
		self.weight = 1.

	
	def __call__(self, x, X):
		return self.bridge(x, X, **self.params)
	

	def nonlinear_params(self):
		nlp = self.params.copy()
		nlp.pop('D')
		return nlp


	def estimate_diffusion_coefficient(self, x, y, BOXES=1):
		X = x[-1]
		x, y = x[1:-1], y[1:-1]
		self.params['D'] = 1.
		f = self(x, X)
		self.params['D'] = diffusion_sum(y, f)
		return self.params['D']


	def chi_2(self, x, X, y):
		return np.sum(self(x[1:-1], X)-y[1:-1])


###


class fluctuation_analysis(object):

	def __init__(self, color='all', verbose=False):
		self.color = color
		self.bridge = bridge_process(color=color)
		self.verbose = verbose



	def checkin_args(self, y, BOX_SIZE, STRIDE=1):
		self.y = np.asarray(y, dtype=float)
		self.BOX_SIZE = BOX_SIZE
		self.STRIDE = STRIDE

		if self.y.size < self.BOX_SIZE:
			raise ValueError

	

	def prepare_data(self):
		self.x_box = np.arange(self.BOX_SIZE, dtype=float)	# x-axis variable
		self.y_mean = bridge_variance(self.y, self.BOX_SIZE, stride=self.STRIDE)

		if self.verbose:
			from pylab import figure
			self.fig = figure()
			self.ax = self.fig.add_subplot(111)
			self.ax.plot(self.x_box, self.y_mean, 'ko-')
			self.li_bridge, = self.ax.plot([], [], 'r--')
			self.p_text = self.ax.text(self.x_box.max()/2, self.y_mean.max()/2, '')
			self.fig.canvas.draw()


	def __str__(self):
		chi_2 = self.bridge.chi_2(self.x_box, self.x_box[-1], self.y_mean)
		string = '$\chi^2=%.3g$\n $\Delta=%.3g$\n' % (chi_2, self.bridge.params['D'])

		for name in self.bridge.nonlinear_params().keys():
			string += '$'+name+'=%.3g$\n' % (self.bridge.params[name])

		return string


	def erf_verbose(self):
		self.li_bridge.set_data(self.x_box, self.bridge(self.x_box, self.x_box[-1]))
		self.p_text.set_text(str(self))
		self.fig.canvas.draw()


	def erf(self, p, names):

		for i in xrange(len(names)):
			self.bridge.params[names[i]] = p[i]

		self.bridge.estimate_diffusion_coefficient(self.x_box, self.y_mean)
		chi_2 = self.bridge.chi_2(self.x_box, self.x_box[-1], self.y_mean)

		if self.verbose:
			self.erf_verbose()

		return chi_2


	def optimize_params(self):
		nlp = self.bridge.nonlinear_params()
		keys = nlp.keys()
		p_opt = [nlp[k] for k in keys]

		if len(nlp):
			bb = []
			for k in keys:
				bb.append(bounds[self.color][k])
			
			p_opt = opt.fmin_l_bfgs_b(func=self.erf, x0=p_opt, args=(keys,), bounds=bb, approx_grad=True)[0]

		else:
			self.erf(p_opt, keys)

		return self.bridge.chi_2(self.x_box, self.x_box[-1], self.y_mean)


	def __call__(self, y, BOX_SIZE=None, STRIDE=1):

		try:
			self.checkin_args(y, BOX_SIZE, STRIDE)	# check and store all arguments
		except:
			return self.bridge.params, LARGE


		self.prepare_data()			# compute the average across y boxes
		chi_2 = self.optimize_params()
		p_opt = self.bridge.params

		return p_opt, chi_2


def pFiDA(y, BOX_SIZE, color='white', STRIDE=1, verbose=False):
	analyser = fluctuation_analysis(color=color, verbose=verbose)
	return analyser(y, BOX_SIZE=BOX_SIZE, STRIDE=STRIDE)


def pFiDA_return_times(return_times, BOX_SIZE, color='white', STRIDE=1, verbose=False):
	y = np.concatenate(([0.], np.cumsum(return_times)))
	return  pFiDA(y, BOX_SIZE=BOX_SIZE, STRIDE=STRIDE, color=color, verbose=verbose)


#=========================================================================================

####################################################
###  and this is an example use of the module.
####################################################

if __name__ == "__main__":


	from pylab import *


	N = 5000 # number of return times

	### construction of return time intervals
	return_time_intervals = randn(N)
	s = fftpack.rfft(return_time_intervals.copy())	
	f = 1./(1.+array(arange(0., s.size, 1.), dtype=float))**0.5
	return_time_intervals = irfft(s*f, len(return_time_intervals))
	return_time_intervals = 10.+return_time_intervals

        figure()
        plot(cumsum(return_time_intervals), return_time_intervals, 'ko-')
        xlabel('return time $t_n$', fontsize=20)
        ylabel('return time interval $t_{n+1}-t_n$', fontsize=20)
        title('Test timeseries', fontsize=20)
        tight_layout()
	### construction of return time intervals


	### phase diffusion analysis of return time intervals using the brownian bridge model
	pFiDA_return_times(return_time_intervals, BOX_SIZE=300, color='white', verbose=True)
        title('Brownian Bridge Model')
        xlabel('time index', fontsize=20)
        ylabel('bridge process variance', fontsize=20)
	### phase diffusion analysis of return time intervals using the brownian bridge model


	### phase diffusion analysis of return time intervals using the colored bridge model (see paper)
	pFiDA_return_times(return_time_intervals, BOX_SIZE=300, color='colored', verbose=True)
        title('Colored Bridge Model (as described in paper)')
        xlabel('time index', fontsize=20)
        ylabel('bridge process variance', fontsize=20)
	### phase diffusion analysis of return time intervals using the brownian bridge model





	show()










