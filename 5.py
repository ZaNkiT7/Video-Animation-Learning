import matplotlib.pyplot as plt #imports the matplotlib library the most important for plotting it provides pyplot module which contains functions for creating various types of plots
import numpy as np
val = np.random.randint(1, 10, 10)
plt.bar(range(len(val)),val)
plt.show() 
print(*val)
print(*val)
print(*val)
while True:
    val = np.random.randint(1,10,10)
    plt.figure(figsize=(10,5))
    plt.errorbar(range(len(val)),val,val,
                 fmt='o',capsize=6)
    plt.plot(range(20),color='red')
    print(*val)
    
    plt.title (label='Title',#labeling of chart
              fontsize=20,# size
              rotation =90,# normally it is center but change angle to 90
              loc='right' ) # right is for moving this to right of screen    
    x=np.arange(0,40,0.1)
    y=np.cos(x)
    plt.plot(x,y)
    plt.show()        