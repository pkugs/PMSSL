import numpy as np
from math import ceil, pow, floor, sqrt
import time

def get_location(c, fs, rr, nMicrophones, nSamples, ss, LL, beta, microphone_type, nOrder, angle, isHighPassFiler):
    rr = rr.reshape(-1)
    L = []
    cTs = c/fs
    for i in range(3):
        L.append(LL[i]/cTs)
    n1 = ceil(nSamples/(2*L[0]))
    n2 = ceil(nSamples/(2*L[1]))
    n3 = ceil(nSamples/(2*L[2]))    

    nLocate = []
    cTs = c / fs
    s = []
    for i in range(3):
        s.append(ss[i])
        # s.append(ss[i]/cTs)

    L = []
    for i in range(3):
        L.append(LL[i])
        # L.append(LL[i]/cTs) 

    for idxMicrophone in range(nMicrophones):
        idxLocate = []
        r = []
        Rm = [0, 0, 0]

        for t in range(3):
            r.append(rr[idxMicrophone + t*nMicrophones])
            # r.append(rr[idxMicrophone + t*nMicrophones]/cTs)
        n1 = ceil(nSamples*cTs/(2*L[0]))
        n2 = ceil(nSamples*cTs/(2*L[1]))
        n3 = ceil(nSamples*cTs/(2*L[2]))

        for mx in range(-n1, n1 + 1):
            Rm[0] = 2*mx*L[0]
            for my in range(-n2, n2 + 1):
                Rm[1] = 2*my*L[1]
                for mz in range(-n3, n3+1):
                    Rm[2] = 2*mz*L[2]
                    Rp_plus_Rm = np.zeros((3, ), dtype=np.float32)
                    for q in range(2):
                        Rp_plus_Rm[0] = (1-2*q)*s[0] - r[0] + Rm[0]
                        for j in range(2):
                            Rp_plus_Rm[1] = (1-2*j)*s[1] - r[1] + Rm[1]
                            for k in range(2):
                                Rp_plus_Rm[2] = (1-2*k)*s[2] - r[2] + Rm[2]
                                dist = sqrt(pow(Rp_plus_Rm[0]/cTs, 2) + pow(Rp_plus_Rm[1]/cTs, 2) + pow(Rp_plus_Rm[2]/cTs, 2))
                                kOther = abs(2*mx-q)+abs(2*my-j)+abs(2*mz-k)
                                if (kOther <= nOrder) or (nOrder == -1):
                                    if floor(dist) < nSamples:
                                        while len(idxLocate) <= kOther:
                                            idxLocate.append([])
                                        idxLocate[kOther].append([Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2]])
                                        # idxLocate.append([Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2]])
        nLocate.append(idxLocate)
    return nLocate


def get_locat(c,samplingRate,micPositions,srcPosition,LL,**kwargs):

    '''
    Input parameters:
    c           : sound velocity in m/s.
    fs          : sampling frequency in Hz.
    r           : M x 3 array specifying the (x,y,z) coordinates of the receiver(s) in m.
    s           : 1 x 3 vector specifying the (x,y,z) coordinates of the source in m.
    L           : 1 x 3 vector specifying the room dimensions (x,y,z) in m.
    ------ Parameters below needs to be specified as: key=value 
    ---------ex: beta=0.4,nsample=4096,mtype='omnidirectional',order=-1,dim=3,orientation=0,hp_filter=1
    beta        : 1 x 6 vector specifying the reflection coefficients
                    [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2] or
                    beta = reverberation time (T_60) in seconds.
    nsample     : number of samples to calculate, default is T_60*fs.
    mtype       : [omnidirectional, subcardioid, cardioid, hypercardioid,bidirectional], default is omnidirectional.
    order       : reflection order, default is -1, i.e. maximum order.
    dim         : room dimension (2 or 3), default is 3.
    orientation : direction in which the microphones are pointed, specified using 
                    azimuth and elevation angles (in radians), default is [0 0].
    hp_filter   : use '0' to disable high-pass filter, the high-pass filter is enabled by default.
    '''
    if type(LL) is not np.array:
        LL=np.array(LL,ndmin=2)
    if LL.shape[0]==1:
        LL=np.transpose(LL)

    if type(micPositions) is not np.array:
        micPositions=np.array(micPositions,ndmin=2)
    if type(srcPosition) is not np.array:
        srcPosition=np.array(srcPosition,ndmin=2)
    
    beta = np.zeros((6, 1), dtype=np.float32)

    if 'beta' in kwargs:
        betaIn=kwargs['beta']
        if type(betaIn) is not np.array:
            betaIn=np.transpose(np.array(betaIn,ndmin=2))
        if (betaIn.shape[1])>1:
            beta=betaIn
            V=LL[0]*LL[1]*LL[2]
            alpha = ((1-pow(beta[0],2))+(1-pow(beta[1],2)))*LL[0]*LL[2]+((1-pow(beta[2],2))+(1-pow(beta[3],2)))*LL[1]*LL[2]+((1-pow(beta[4],2))+(1-pow(beta[5],2)))*LL[0]*LL[1]
            reverberation_time = 24*np.log(10.0)*V/(c*alpha)
            if (reverberation_time < 0.128):
                reverberation_time = 0.128
        else:
            reverberation_time=betaIn		
            if (reverberation_time != 0) :
                V=LL[0]*LL[1]*LL[2]
                S = 2*(LL[0]*LL[2]+LL[1]*LL[2]+LL[0]*LL[1])		
                alfa = 24*V*np.log(10.0)/(c*S*reverberation_time)
                if alfa>1:
                    raise ValueError("Error: The reflection coefficients cannot be calculated using the current room parameters, i.e. room size and reverberation time.\n Please specify the reflection coefficients or change the room parameters.")
                beta=np.zeros([6,1])
                beta+=np.sqrt(1-alfa)
            else:
                beta=np.zeros([6,1])
    else:
            raise ValueError("Error: Specify either RT60 (ex: beta=0.4) or reflection coefficients (beta=[0.3,0.2,0.5,0.1,0.1,0.1])")
    
    """Number of samples: Default T60 * Fs"""
    if 'nsample' in kwargs:
        nsamples=kwargs['nsample']
    else:
        nsamples=int(reverberation_time * samplingRate)

    """Mic type: Default omnidirectional"""
    m_type='omnidirectional'
    if 'mtype' in kwargs:
        m_type=kwargs['mtype']
    if m_type is 'bidirectional':
        mtype = 'b'
    if m_type is 'cardioid':
        mtype = 'c'
    if m_type is 'subcardioid':
        mtype = 's'
    if m_type is 'hypercardioid':
        mtype = 'h'
    if m_type is 'omnidirectional':
        mtype = 'o'	

    """Reflection order: Default -1"""
    order = -1
    if 'order' in kwargs:
        order = kwargs['order']
        if order<-1:
            raise ValueError("Invalid input: reflection order should be > -1")

    """Room dimensions: Default 3"""
    dim=3
    if 'dim' in kwargs:
        dim=kwargs['dim']
        if dim not in [2,3]:
            raise ValueError("Invalid input: 2 or 3 dimensions expected")
        if dim is 2:
            beta[4]=0
            beta[5]=0

    """Orientation"""
    angle = np.zeros((2,1), dtype=np.float32)
    if 'orientation' in kwargs:
        orientation=kwargs['orientation']
        if type(orientation) is not np.array:
            orientation=np.array(orientation, ndmin=2)
        if orientation.shape[1]==1:
            angle[0]=orientation[0]
        else:
            angle[0]=orientation[0,0]

            angle[1]=orientation[0,1]

    """hp_filter enable"""
    isHighPassFilter=1
    if 'hp_filter' in kwargs:
        isHighPassFilter=kwargs['hp_filter']

    numMics=micPositions.shape[0]

    return get_location(c, samplingRate, np.ascontiguousarray(np.transpose(micPositions)), numMics, nsamples, np.ascontiguousarray(np.transpose(srcPosition)), np.ascontiguousarray(LL), beta, mtype, order, angle, isHighPassFilter)   


if __name__ == '__main__':
    c = 344						# Sound velocity (m/s)
    fs = 16000					# Sample frequency (samples/s)
    r = [[2,1.5,2]]	# Receiver position [x y z] (m)
    s = [2.4,3.7,2]				# Source position [x y z] (m)
    L = [5,4,6]					# Room dimensions [x y z] (m)
    beta = 0.4					# Reflections Coefficients
    n = 5000					# Number of samples
    mtype = 'omnidirectional'	# Type of microphone
    order = 3   				# Reflection order
    dim = 3						# Room dimension
    orientation = 0				# Microphone orientation (rad)
    hp_filter = 1				# Enable high-pass filter
    start = time.clock()
    loc = get_locat(c, fs, r, s, L, beta=beta, nsample=n, mtype=mtype, dim=dim)
    for ii in range(len(r)):
        for jj in range(order):
            for yy in range(len(loc[ii][jj])):
                if np.abs(loc[ii][jj][yy][2])<0.1:
                    print(loc[ii][jj][yy])
            print("next order")
    elapsed = (time.clock()-start)
    print('time:',elapsed)
    # print(loc[0][0])
