#!/usr/bin/env python
import os

#Import stuff
import numpy as np 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from astropy.io import fits

#import seaborn as sns
rcParams['font.family'] = 'serif'

# Adjust rc parameters to make plots pretty
def plot_pretty(dpi=200, fontsize=8):
    
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)       # dpi resolution of saved image files
    # if you have LaTeX installed on your laptop, uncomment the line below for prettier labels
    plt.rc('text', usetex=True)      # use LaTeX to process labels
    plt.rc('font', size=fontsize)    # fontsize
    plt.rc('xtick', direction='in')  # make axes ticks point inward
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=10) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=10) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1]) # fix dotted lines

    return

plot_pretty()


from matplotlib import gridspec
from scipy.stats import gaussian_kde

def jointPlot(data, dims,cols,bins,kde=False,**kwargs):
    '''
    data = our dataset - dimensions (:,2) or (:,4) depending on the dimension
    dims = 2 or 4 - 2 in case we have a joint plot of one type of data - 4 if we have two types of data
    cols = colors (1 or 2)
    '''
    # ============================================================
    # ============================================================
    # Define the max and mins of the first dataset
    x_min_1, x_max_1 = data[:,0].min(),data[:,0].max()
    y_min_1, y_max_1 = data[:,1].min(),data[:,1].max()
    
    # Now if dims = 4, find the min and max of the second dataset as well
    if (dims==4):
        x_min_2, x_max_2 = data[:,2].min(),data[:,2].max()
        y_min_2, y_max_2 = data[:,3].min(),data[:,3].max()
    
    # ============================================================
    # ============================================================
    # Define grid for subplots
    gs = gridspec.GridSpec(2, 2,wspace=0.2,hspace=0.2, width_ratios=[4, 1], height_ratios = [1, 4])
    
    # ============================================================
    # ============================================================
    #Create scatter plot
    fig = plt.figure(figsize=(5.5,5.5),facecolor='white')
    ax = plt.subplot(gs[1, 0],frameon = True)
    cax = ax.scatter(data[:,0], data[:,1],rasterized=True, color=cols[0], s=0.5, alpha=.6)
    # Now in case dims=4, add one more scatter plot
    if (dims==4):
        cax = ax.scatter(data[:,2], data[:,3], rasterized=True,color=cols[1], s=0.5, alpha=.6)
   
    ax.grid(ls='--', axis='both' ,alpha=0.6)
    
    ax.set_xlabel(kwargs['xlabel'],fontsize=11)
    ax.set_ylabel(kwargs['ylabel'],fontsize=11)
    # ===============================================================
    # ===============================================================
    # Lower and upper limits in the x and y directions
    x_low = kwargs['xlow']
    x_up = kwargs['xup']
    y_low = kwargs['ylow']
    y_up = kwargs['yup']
    # ===============================================================
    # ===============================================================
    #Create Y-marginal (right)
    axr = plt.subplot(gs[1, 1], sharey=ax, frameon = True, xticks = [],ylim=(y_low,y_up)) 
    axr.hist(data[:,1],bins=bins, color = cols[0],alpha=0.6, orientation = 'horizontal', normed = True)
    # In case dims = 4, add one more historgram
    if (dims==4):
        axr.hist(data[:,3],bins=bins, color = cols[1],alpha=0.6, orientation = 'horizontal', normed = True)
        
    
    axr.grid(ls='--', axis='both' ,alpha=0.6)
    
    
    # ===============================================================
    #Create X-marginal(top)
    axt = plt.subplot(gs[0,0], sharex=ax,frameon = True, yticks=[],xlim=(x_low,x_up))
    axt.hist(data[:,0],bins=bins, color = cols[0],alpha=0.6, normed = True)
    # In case dims = 4, add one more histogram
    if (dims==4):
        axt.hist(data[:,2],bins=bins, color = cols[1],alpha=0.6, normed = True)
        
    axt.grid(ls='--', axis='both' ,alpha=0.6)
    
    #Bring the marginals closer to the scatter plot
    fig.tight_layout(pad = 0.0)

    if kde:
        kdex_1=gaussian_kde(data[:,0])
        kdey_1=gaussian_kde(data[:,1])
        x_1= np.linspace(x_min_1,x_max_1,100)
        y_1= np.linspace(y_min_1,y_max_1,100)
        dx_1=kdex_1(x_1)
        dy_1=kdey_1(y_1)
        axr.plot(dy_1,y_1,color='k',linewidth=1)
        axt.plot(x_1,dx_1,color='k', linewidth=1)
        
        # And in case dims = 4, we have more kdes
        if (dims==4):
            kdex_2=gaussian_kde(data[:,2])
            kdey_2=gaussian_kde(data[:,3])
            x_2= np.linspace(x_min_2,x_max_2,100)
            y_2= np.linspace(y_min_2,y_max_2,100)
            dx_2=kdex_2(x_2)
            dy_2=kdey_2(y_2)
            axr.plot(dy_2,y_2,color='black', ls='--')
            axt.plot(x_2,dx_2,color='black', ls='--')
        
    return ax,axt,axr
    

# =====================================================================================
# =====================================================================================

def color_plot(mag_g,mag_r,mag_i):
    # Define the colors here
    col_g_i = mag_g - mag_i
    col_g_r = mag_g - mag_r

    # Calculate and print the median of the g-i color
    med_g_i = np.nanmedian(col_g_i)

    data = np.zeros((len(col_g_i[(col_g_i>-1.0)&(col_g_r<1.5)]),2))
    data[:,1] = col_g_r[(col_g_i>-1.0)&(col_g_r<1.5)]
    data[:,0] = col_g_i[(col_g_i>-1.0)&(col_g_r<1.5)]

    bins = np.linspace(-0.5,2.0,75)
    #ax,axt,axr = jointPlot(data, dims=2,cols=['g'],bins=50,kde=True,xlabel='$g-i$', ylabel='$g-r$', xlow=-0.2,xup=1.5,ylow=-0.2,yup=1.2)
    ax,axt,axr = jointPlot(data, dims=2,cols=['g'],bins=bins,kde=True,xlabel='$g-i$', ylabel='$g-r$', xlow=-0.2,xup=1.5,ylow=-0.2,yup=1.2)

    ax.vlines(med_g_i,-0.25,1.25, color='k', linewidth=0.8,linestyle='--')
    ax.text(0.2,0.81, 'Blue \n LSBGs', color='blue',fontsize=12)
    ax.text(1.0,0.12, 'Red \n LSBGs', color='red',fontsize=12)

    axr.tick_params(axis='both', labelsize=10)
    axt.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='both', labelsize=11)

    print('megian g-i color is:')
    print(med_g_i)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename')
    args = parser.parse_args()

    data = fits.open(args.filename)[1].data

    #sel = np.abs(data['RE']/data['FLUX_RADIUS_I'] - 1) < 0.1
    #ra,dec = 54.6, -35.4
    ra,dec = 30.0, -50.0
    ra,dec = 21.4, -1.4
    delta = 5

    #sel = (np.abs(data['RA'] - delta) < 10) & (np.abs(data['DEC'] - dec) < delta)
    #sel = np.abs(data['RA'] - 35) < 3
    #sel = slice(0000,1000)
    #data = data[sel]

    color_plot(data['MAG_G'],data['MAG_R'],data['MAG_I'])
    title = os.path.splitext(os.path.basename(args.filename))[0]
    plt.suptitle(title,usetex=False)
    plt.savefig('color-color.png')

    plt.figure()
    plt.subplot(111, projection="mollweide")
    ra = np.radians(data['RA']-360*(data['RA']>180))
    dec = np.radians(data['DEC'])
    plt.scatter(ra,dec,s=2,alpha=0.2)
    plt.grid(True)
    plt.show()
