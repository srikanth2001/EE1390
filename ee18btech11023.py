#by K Srikanth

import numpy as np
import matplotlib.pyplot as plt


#if using termux
import subprocess
import shlex
#end if

data=np.loadtxt('ee18btech11023.dat')
plt.plot(data[:,0],data[:,1])
plt.grid()
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title('Spice simulation for feedback amplifier')
#plt.show()
#if using termux



plt.savefig('../../figs/EE18BTECH11023/ee18btech11023_output.pdf')
plt.savefig('../../figs/EE18BTECH11023/ee18btech11023_output.eps')
subprocess.run(shlex.split("termux-open ./figs/EE18BTECH11023/ee18btech11023_output.pdf"))
#else
#plt.show()
