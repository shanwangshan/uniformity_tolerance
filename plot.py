import numpy as np
from matplotlib import pyplot as plt

t = np.arange(0.1,1.1,step=0.1)

# uniformity

baseline = np.array([0.3976,0.3719,0.2954,0.3126,0.3377,0.3106,0.3166,0.3129,0.2680,0.3560])
ACL= np.array([0.2447,0.2605,0.1874,0.1733,0.1944,0.1839,0.2118,0.1512,0.2052,0.1530])


plt.figure(figsize=(13,7.5))


plt.plot(t,baseline,'-s',label = 'Contrastive loss',linewidth=3,markerfacecolor='red',markersize=12)
plt.plot(t,ACL,'-D',label='ACL (ours)',linewidth=3,markersize=12,markerfacecolor='blue')
#plt.title('Uniformity',fontsize=20)
plt.xlabel(r'Temperature ($\tau$)',fontsize=25)
plt.ylabel('-U', fontsize=25)

plt.xticks(np.arange(0.1, 1.1, step=0.1),fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.grid()
plt.savefig("Uniformity.pdf")

# tolerance

# baseline = np.array([0.8080,0.7282,0.8113,0.8997,0.8386,0.8665,0.8858,0.8935,0.9324,0.9164])
# aml = np.array([0.9385,0.9271,0.9417,0.9506,0.9405,0.9491,0.9276,0.9542,0.9439,0.9538])
baseline = np.array([0.9113,0.9189,0.9357,0.9328,0.9269,0.9339,0.9301,0.9329,0.9440,0.9271])
ACL = np.array([0.9471,0.9437,0.9437,0.9632,0.9589,0.9601,0.9532,0.9684,0.9595,0.9676])
plt.figure(figsize=(13,7.5))
plt.plot(t,baseline,'-s',label = 'Contrastive loss',linewidth=3,markersize=12,markerfacecolor='red')
plt.plot(t,ACL,'-D',label = 'ACL (ours)',linewidth=3,markersize=12,markerfacecolor='blue')
#plt.title('Tolerance',fontsize=20)
plt.xlabel(r'Temperature ($\tau$)',fontsize=25)
plt.ylabel('T', fontsize=25)

plt.xticks(np.arange(0.1, 1.1, step=0.1),fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.grid()

plt.savefig("Tolerance.pdf")

# linear probe accuracy
baseline = np.array([77.081,74.160,75.570,75.175,74.893,75.626,75.897,74.126,72.930,71.329])
ACL = np.array([78.356,77.070,75.615,75.389,77.149,77.645,76.664,75.276,76.810,75.964])
plt.figure(figsize=(13,7.5))
plt.plot(t,baseline,'-s',label = 'Contrastive loss',linewidth=3,markersize=12,markerfacecolor='red')
plt.plot(t,ACL,'-D',label = 'ACL (ours)',linewidth=3,markersize=12,markerfacecolor='blue')
#plt.title('Downstream classification accuracy',fontsize=20)
plt.xlabel(r'Temperature ($\tau$)',fontsize=25)
plt.ylabel('Accuracy (%)',fontsize=25)
plt.xticks(np.arange(0.1, 1.1, step=0.1),fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.grid()

plt.savefig("Linear_probe.pdf")
