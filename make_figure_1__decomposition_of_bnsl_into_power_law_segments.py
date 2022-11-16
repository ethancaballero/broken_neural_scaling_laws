import numpy as np
import matplotlib.pyplot as plt

x_min = -1
x_max = 26
x = np.logspace(x_min, x_max, 8192)

# unimprovable performance
a = 1e-3

# offset (on log-log plot)
b = 4.66e3

#changes in slope (on log-log plot)
c0 = 0.05
c1 = .49
c2 = -0.811
c3 = 2.1

#where breaks happens
d1 = 3e6
d2 = 1e14
d3 = 2e20

# sharpness of transitions during breaks; smaller (nonnegative) values are sharper; larger values are less sharp
f1 = 1.1
f2 = 1.425
f3 = .3
f3 = 0.05

"""
Decomposition based on combination of insights from: 
https://en.wikipedia.org/wiki/Power_law#Broken_power_law , 
https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D.html , 
paragraph below equation 1.1 of http://wallpaintings.at/geminga/Multiply_broken_power-law_densities_as_survival_functions.pdf#page=2
"""

# broken neual scaling law (bnsl) with 3 breaks
y = a + b*(x)**(-c0) * (1.+(x/d1)**(1./f1))**((-c1)*f1) * (1.+(x/d2)**(1./f2))**((-c2)*f2) * (1.+(x/d3)**(1./f3))**((-c3)*f3)

x1 = x[(x <= d1*25)]
x2 = x[(x >= d1*.04) & (x <= d2*25)]
x3 = x[(x >= d2*.04) & (x <= d3*25)]
x4 = x[(x >= d3*.32)]

#individual power law segments within the bnsl
segment1 =      b * (x1)**(-c0)
segment2 =      b * (d1)**(-(c0)) * (x2/d1)**(-(c1+c0))
segment3 =      b * (d1)**(-(c0)) * (d2/d1)**(-(c1+c0)) * (x3/d2)**(-(c2+c1+c0))
segment4 =      b * (d1)**(-(c0)) * (d2/d1)**(-(c1+c0)) * (d3/d2)**(-(c2+c1+c0)) * (x4/d3)**(-(c3+c2+c1+c0))
#segment4 = a + b * (d1)**(-(c0)) * (d2/d1)**(-(c1+c0)) * (d3/d2)**(-(c2+c1+c0)) * (x4/d3)**(-(c3+c2+c1+c0))
linewidth = 2.0

plt.figure(figsize=(6.4, 4))

plt.title("Decomposition of BNSL into Power Law Segments")

plt.plot(x, y, color = 'black', label='BNSL', linewidth=3.5)

plt.axvline(x = d1, linestyle=':', color = [0.8,0.0,0.8], label = 'Break 1', linewidth=linewidth)
plt.axvline(x = d2, linestyle=':', color = [0.6,0.0,0.6], label = 'Break 2', linewidth=linewidth)
plt.axvline(x = d3, linestyle=':', color = [0.4,0.0,0.4], label = 'Break 3', linewidth=linewidth)

plt.plot(x1, segment1, '--', label='Segment 1', color = [0.8,0.8,0.0], linewidth=linewidth)
plt.plot(x2, segment2, '--', label='Segment 2', color = [0.0,0.9,0.9], linewidth=linewidth)
plt.plot(x3, segment3, '--', label='Segment 3', color = [1.0, 0.45, 0.45], linewidth=linewidth)
plt.plot(x4, segment4, '--', label='Segment 4', color = [0.2, 0.925, 0.2], linewidth=linewidth)

plt.axhline(a, linestyle=('-.'), color = 'silver', label = 'Limit', linewidth=linewidth*.89)

plt.xlabel("Quantity Being Scaled")
plt.ylabel("Performance Evaluation Metric")

plt.xscale('log')
plt.yscale('log')

plt.xlim(1.1*(10**x_min), .9*(10**x_max))
plt.ylim(y.min()/1.5, y.max()*1.5)

plt.legend(loc='lower left')

plt.savefig('figure_1.png', bbox_inches='tight')
plt.show()

plt.close()
plt.cla()
plt.clf()
