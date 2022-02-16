import numpy as np
def closenessfinder(lseg):#Function to find the closest power of 2 to the segment size
	pow2arr=[2**i for i in range (0,30)] #Array of powers of 2, cut off at 2**30
	disarr=[lseg-i for i in pow2arr] 
	newarr=[i if i>=0 else abs(2**30) for i in disarr]#Checks min distance of len from 2**x
	return(pow2arr[np.argmin(newarr)])
def powerspec(ts,fs=1, ns=1, lseg=None):
	if(lseg==None): #If segment length is not defined
		lseg=closenessfinder(len(ts)/ns)
	elif(ns==1):#If number of segments is not defined
		ns=len(ts)//lseg
	leng=int(lseg*ns)
	ts=ts[0:leng]
	ts=[i-np.mean(ts) for i in ts]#Removing DC component
	ps=[]
	for i in range(0,ns):
		 ps.append(abs(np.fft.fft(ts[(i*int(leng/ns)):((i+1)*int(leng/ns))]))**2) #Powerspectrum array #Add options for windows?np.hamming(leng/ns)
	psa=np.zeros(len(ps[0]))	
	for i in range (0,ns):
		psa=psa+np.array(ps[i])
	freq = [i*fs/len(psa) for i in range(0,int(len(psa)/2))] # Frequencies
	dt=1./fs
	psa=psa*dt
	psa=psa/ns              #Normalisation
	return(freq,psa[0:len(freq)]) 	#Returns frequencies and power spectrum
	

def maxpeak(ts):#Returns frequency of the maximum peak
	ps_t=powerspec(ts)
	return(ps_t[0][np.argmax(ps_t[1])])

