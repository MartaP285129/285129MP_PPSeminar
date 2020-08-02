
import matplotlib.pyplot as plt
import numpy as np
import mdshare
import math
import argparse
import textwrap
import os.path
import yaml
import warnings


from sklearn.manifold import TSNE
from scipy import stats



# Density estimation 
def density_estimation(m1, m2):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z
# Converting high-dimensional tensor and returns its projection onto a low-dimensional space.
def fit(Y):
    global xmin,xmax,ymin,ymax
    if V_pca:
        Y_TSNE = TSNE(n_components=2,learning_rate=V_learning_rate,n_iter=V_n_iter,min_grad_norm=V_min_grad_norm).fit_transform(Y[::V_sampling,:])
    else:
        Y_TSNE = TSNE(n_components=2,learning_rate=V_learning_rate,n_iter=V_n_iter,min_grad_norm=V_min_grad_norm, init='pca').fit_transform(Y[::V_sampling,:])
# Linear interpolation
    Y_norm=np.interp(Y_TSNE[:,0:2], (Y_TSNE[:,0:2].min(), Y_TSNE[:,0:2].max()), (-math.pi, +math.pi))
# Density estimation
    xmin = Y_norm[:,0].min()
    xmax = Y_norm[:,0].max()
    ymin = Y_norm[:,1].min()
    ymax = Y_norm[:,1].max()
    x, y, z = density_estimation(Y_norm[:,0], Y_norm[:,1])
# Create figure
    fig, ax = plt.subplots()
# Show density 
    ax.imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
# Add contour lines
    plt.contour(x, y, z) 
# Show points                                                                          
    if V_visible_points:
        ax.plot(Y_norm[:,0], Y_norm[:,1], 'k.', markersize=2)    
    ax.set_xlim([xmin, xmax])                                                                           
    ax.set_ylim([ymin, ymax])                                                                           
    plt.show()

def Main():
    global V_sampling,V_learning_rate,V_n_iter,V_min_grad_norm,V_visible_points,V_pca, args
    
# Setting variables from console
    parser = argparse.ArgumentParser(
         prog='project',
         formatter_class=argparse.RawDescriptionHelpFormatter,
         description=textwrap.dedent('''\
             Marta Preis (285129)
             ----------------------------------------
                 Converting high-dimensional tensor
                 and returns its projection onto
                 a low-dimensional space.
             ----------------------------------------
             '''))
    parser.add_argument('-d','--data', metavar='', default='alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
                        help='Path do data or name (download using mdshare)')
    parser.add_argument('-s','--sampling',metavar='', type=int, default=500, 
                        help='Using only every x sample. (default: 500)')
    parser.add_argument('-l','--learning_rate',metavar='', type=int, default=200, 
                        help='The learning rate is usually in the range [10.0, 1000.0]. (default: 200)')
    parser.add_argument('-i','--n_iter', metavar='',type=int, default=1000, 
                        help='Maximum number of iterations for the optimization. Should be at least 250. (default: 1000)')
    parser.add_argument('-n','--min_grad_norm', metavar='', type=float, default=1e-5, 
                        help='If the gradient norm is below this threshold, the optimization will be stopped. (default: 1e-5)')
    parser.add_argument('-v','--visible_points', action='store_true', 
                        help='Show all points in plot')
    parser.add_argument('-p','--pca', action='store_true', 
                        help='Initialization of embedding pca')
# Show help
    parser.print_help()
# Set variables
    args = parser.parse_args()
    V_data=args.data
    V_sampling=args.sampling
    V_learning_rate=args.learning_rate
    V_n_iter=args.n_iter
    V_min_grad_norm=args.min_grad_norm
    V_visible_points=args.visible_points
    V_pca=args.pca

# Check if file is available, if not, download data 
    if os.path.isfile(V_data):
        dataset = yaml.safe_load(V_data)
    else:
        dataset = mdshare.fetch(V_data)
    with np.load(dataset) as f:
        Y = np.vstack([f[key] for key in sorted(f.keys())])
# Fitting function
    fit(Y)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    Main()

