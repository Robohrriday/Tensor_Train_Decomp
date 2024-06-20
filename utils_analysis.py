import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =  pd.read_csv('compression_ratios_int(0,20)_100x30.csv')
plt.title('compression_ratios_int(0,20)_100x30.csv\nCompression Ratio vs Steady State Relative Error')
plt.plot(data['compression_ratio'], data['steady_state_relative_loss'])
for i in range(len(data)):
    plt.text(data['compression_ratio'][i], data['steady_state_relative_loss'][i], str(data['rank'][i]), fontsize = 8)
plt.xlabel('Compression Ratio')
plt.ylabel('Steady State Relative Error')
plt.grid(alpha = 0.5)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()