from package.integrator import sampling_integrator as sint
from package.convergence_test import test_plane_pulse as tpl
import matplotlib.pyplot as plt   

tpl()
tpl(c=0.75, clr = 'tab:orange')
tpl(c=0.5, clr = 'tab:green')
tpl(c=0.25, clr = 'tab:red')

# plt.figure()
# plt.savefig('t=1.0_tenpercent.pdf')


plt.close()
plt.close()
plt.close()
plt.close()


tpl()
tpl(c=0.75,a1 = 0.1/0.75, clr = 'tab:orange')
tpl(c=0.5, a1 = 0.1/0.5, clr = 'tab:green')
tpl(c=0.25,a1 = 0.1/0.25, clr = 'tab:red')

# plt.savefig('t=1.01tenth.pdf')

plt.close()
plt.close()
plt.close()
plt.close()




tpl(tfinal = 5)
tpl(tfinal = 5, c=0.75, clr = 'tab:orange')
tpl(tfinal = 5, c=0.5, clr = 'tab:green')
tpl(tfinal = 5, c=0.25, clr = 'tab:red')

# plt.savefig('t=5.0_tenpercent.pdf')


plt.close()
plt.close()
plt.close()
plt.close()


tpl(tfinal = 5)
tpl(tfinal = 5, c=0.75,a1 = 0.1/0.75, clr = 'tab:orange')
tpl(tfinal = 5, c=0.5, a1 = 0.1/0.5, clr = 'tab:green')
tpl(tfinal = 5, c=0.25,a1 = 0.1/0.25, clr = 'tab:red')

# plt.savefig('t=5.01tenth.pdf')

plt.close()
plt.close()
plt.close()
plt.close()