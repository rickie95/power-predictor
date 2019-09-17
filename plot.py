import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

"""
    Use this script to visualize results from a previous execution of Prophet. 
    
    Useful tips:
        - The results data-frame often contains all the series reconstructed, if you want to focus only to the missing 
          interval consider plotting using python array selection: array[ start_index : end_index ]
          
        - If your're dealing with a lot of data use plot controls to zoom and pan across.
        
        - Matplotlib can save in a multitude of format: pdf, svg, png. Avoid compressed formats if you intend attach
          your plots on a document or print.
"""

original_data = pd.read_csv(os.path.join("input", "dataset.csv"))
results_data = pd.read_csv("prediction.csv")

plt.figure(figsize=(50, 10))  # Size is in... inches. ¯\_(ツ)_/¯

results_data['ds'] = pd.to_datetime(results_data['ds'])

plt.margins(0, 0, tight=True)

plt.plot(results_data['ds'][:-12*60], original_data['LHO.W1'], 'black')
plt.plot(results_data['ds'], results_data['yhat'], 'blue')
plt.plot(results_data['ds'], results_data['yhat_upper'], 'red')
plt.plot(results_data['ds'], results_data['yhat_lower'], 'red')

black_patch = mpatches.Patch(color='black', label='Original data')
blue_patch = mpatches.Patch(color='blue', label='Restored data')
red_patch = mpatches.Patch(color='red', label='Uncertainty range')

plt.legend(handles=[black_patch, blue_patch, red_patch])

plt.subplots_adjust(left=0.04, right=0.97, bottom=0.04, top=0.97)

plt.show()

# plt.savefig("plot.pdf")  # Uncomment to save on file
