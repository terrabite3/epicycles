import matplotlib
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
# from IPython.display import HTML
import numpy as np
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
from copy import deepcopy
from scipy.spatial import distance
from scipy.interpolate import UnivariateSpline
from copy import deepcopy

# Default figure size in notebook
matplotlib.rcParams['figure.figsize'] = (6,6)
matplotlib.rcParams['image.aspect'] = 'equal'







class ImageObject:
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.img = Image.open(BytesIO(response.content))
        self.og_size = self.img.size
    
    def show(self):
        imshow(np.asarray(self.img))

    def monochrome(self, scale=3, threshold=200):
        
        # convert image to monochrome
        image = self.img.convert('L')
        image_array = np.array(image)
        
        # Binarize a numpy array using threshold as cutoff
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if image_array[i][j] > threshold:
                    image_array[i][j] = 255
                else:
                    image_array[i][j] = 0
        
        image = Image.fromarray(image_array)
        
        # scale image down to reduce number of non-zero pixels
        img_sm = image.resize(tuple([int(v/scale) for v in image.size]),Image.ANTIALIAS)
        
        # convert image to black and white
        img_bw = img_sm.convert(mode='1', dither=2) 
        self.bw_img = img_bw
        self.pixels = (1 - np.asarray(img_bw).astype(int))
        self.pixels_flat = np.reshape(self.pixels, self.pixels.size)
    
    def show_bw(self):
        print("Dimensions: {}".format(self.bw_img.size))
        print("Num. pixels: {}".format(self.pixels.sum()))
        imshow(np.asarray(self.bw_img))
    
    def get_tour(self, starting_point="random", plot=True):
        # Get greedy tour through pixels
        
        absolute_index = np.where(self.pixels_flat > 0)[0] # positions of non-zero pixels
        relative_index = np.array(range(1, len(absolute_index)+1 ))
        
        # Replace each non-zero pixel in the array with its number
        # i.e., the 10th non-zero pixel will have 10 in its place
        flat_img_mod = deepcopy(self.pixels_flat)
        for rel, pix in enumerate(absolute_index):
            flat_img_mod[pix] = rel+1
        
        # Get coordiantes for each non-zero pixel
        img_idx = np.reshape(flat_img_mod, self.pixels.shape)
        self.coord_list = []
        for p1 in relative_index:
            p1_coords = tuple([int(c) for c in np.where(img_idx==p1)])
            self.coord_list.append(list(p1_coords))
        
        # Calcualte distance between each pair of coords
        dist_mat = distance.cdist(self.coord_list, self.coord_list, 'euclidean')

        # Initialize search space with nearest neighbor tour
        cities = self.coord_list
        num_cities = len(cities)
        if starting_point=="random":
            start = int(np.random.choice(range(num_cities),size=1))
        else:
            assert starting_point < num_cities
            start = starting_point
        tour = [start]
        active_city = start
        for step in range(0, num_cities):
            dist_row = deepcopy(dist_mat[active_city,:])
            for done in tour:
                dist_row[done] = np.inf
            nearest_neighbor = np.argmin(dist_row)
            if nearest_neighbor not in tour:
                tour.append(nearest_neighbor)
            active_city = nearest_neighbor

        y_tour = -np.array([cities[tour[i % num_cities]] for i in range(num_cities+1) ])[:,0]
        y_tour = y_tour - y_tour[0]#- min(y_tour)
        x_tour = np.array([cities[tour[i % num_cities]] for i in range(num_cities+1) ])[:,1]    
        x_tour = x_tour - x_tour[0]#- min(x_tour)

        # Circle tour back to beginning
        np.append(x_tour, x_tour[0])
        np.append(y_tour, y_tour[0])
        num_cities = num_cities + 1
    
        self.x_tour = x_tour
        self.y_tour = y_tour
        self.num_pixels = num_cities
        
        if plot:
            plt.plot(self.x_tour, self.y_tour)
            
    def get_splines(self, degree=1, plot=True):
        # Convert tours into parametric spline curves
        
        x_spl = UnivariateSpline(list(range(0,self.num_pixels)), self.x_tour, k=degree)
        y_spl = UnivariateSpline(list(range(0,self.num_pixels)), self.y_tour, k=degree)
        
        self.x_spl = x_spl
        self.y_spl = y_spl
        
        if plot:
            p = plt.plot(*zip(*[(x_spl(v), y_spl(v)) for v in np.linspace(0, self.num_pixels-1, 1000)]))

            
    def plot_parametric(self, num_points=1000):
        # num_points = number of points at which to sample the curves
        t_vals, x_vals = zip(*[
            (v, self.x_spl(v)) for v in np.linspace(0, self.num_pixels, num_points)
        ])
        x_vals = np.array(x_vals)
        y_vals = np.array([self.y_spl(v) for v in np.linspace(0, self.num_pixels, num_points)])
        t_vals = np.array(t_vals)

        plt.plot(t_vals, x_vals)
        plt.plot(t_vals, y_vals)









class FourierTransform:
    """Calculate the complex Fourier coefficients for a given function
    """
    def __init__(self,
            fxn,  # Function to be transformed (as Python function object)
             # Note: y is parameterized by its own index
             # 13th value in array = value of function at t=12
            rnge, # (.,.) tuple of range at which to evaluate fxn
            N=500,  # Number of coefficients to calculate
            period=None,  # If different than full length of function
            num_points=1000, # Number of points at which to evalute function
            num_circles=50 # This is needed to calculate proper offsets
        ):
        
        self.num_circles = num_circles
        
        t_vals, y = zip(*[(v, fxn(v)) for v in np.linspace(rnge[0], rnge[1]-1, num_points)])
        t_vals = np.array(t_vals)        
        self.t_vals = t_vals
        
        
        # Save the original coords when plotting
        y = np.array(y)
        y = y - y[0]
        self.fxn_vals = np.array(deepcopy(y))
        
        # Spline function doesn't make endpoints exactly equal
        # This sets the first and last points to their average
        endpoint = np.mean([y[0], y[-1]])
        y[0] = endpoint
        y[-1] = endpoint
        
        # Transform works best around edges when function starts at zero
        # (Can't figure out how to avoid Gibbs-type phenomenon when 
        #  intercept !=0 )
        y = y - y[0]
        
        self.N = N
        if period==None:
            period = rnge[1]
        self.period = period
            
        def cn(n):
            c = y*np.exp(-1j*2*n*np.pi*t_vals/period)
            return(c.sum()/c.size)

        coefs = [cn(i) for i in range(1,N+1)]
        self.coefs = coefs
        self.real_coefs = [c.real for c in self.coefs]
        self.imag_coefs = [c.imag for c in self.coefs]
        
        self.amplitudes = np.absolute(self.coefs)
        self.phases = np.angle(self.coefs)
        
        
        def f(x, degree=N):
            # Evaluate the function y at time t using Fourier approximiation of degree N
            f = np.array([2*coefs[i-1]*np.exp(1j*2*i*np.pi*x/period) for i in range(1,degree+1)])
            return(f.sum())
        
        # Evaluate function at all specified points in t domain
        fourier_approximation = np.array([f(t, degree=N).real for t in t_vals])
        circles_approximation = np.array([f(t, degree=self.num_circles).real for t in t_vals])
        
        # Set intercept to same as original function
        #fourier_approximation = fourier_approximation - fourier_approximation[0] + self.original_offset 
        
        # Adjust intercept to minimize distance between entire function, 
        # rather than just the intercepts. Gibbs-type phenomenon causes
        # perturbations near endpoints of interval
        fourier_approximation = fourier_approximation - (fourier_approximation - self.fxn_vals).mean()
        
        circles_approximation = circles_approximation - (circles_approximation - self.fxn_vals).mean()
        self.circles_approximation = circles_approximation
        
        # Origin offset
        self.origin_offset = fourier_approximation[0] - self.fxn_vals[0]
        
        # Circles offset
        self.circles_approximation_offset = circles_approximation[0] - self.fxn_vals[0]
        
        # Set intercept to same as original function
        self.fourier_approximation = fourier_approximation








# Initialize an ImageObject instance and view the Image
# Obviously, it helps to have a simple image with only 2 colors and 
# close to a single path
url = "http://thevirtualinstructor.com/images/continuouslinedrawinghorse.jpg"
horse = ImageObject(url)
# horse.show()


# Scale the image down by a factor of 3 
# and binarize the pixels using a threshold of
# 200 (out of the 0-255 scale) as my cutoff
horse.monochrome(scale=3, threshold=200)
# horse.show_bw()

# Get a tour through the coordinate space of pixels
# Experiment with various starting points until the
# final result looks reasonably simple without too
# many jumps
horse.get_tour(starting_point=1300, plot=False)

# Get parametric spline functions of the tour
horse.get_splines(plot=False)

# Plot the x/y splines as parametric functions
# of a separate varaible, t
# horse.plot_parametric()



# Calculate Fourier approximations; initialize NC equal to the
# number of circles the final animation will have to get an 
# accurate view of how well the image will be approximated
NC = 50
xFT = FourierTransform(horse.x_spl, (0, horse.num_pixels), num_circles=NC)
yFT = FourierTransform(horse.y_spl, (0, horse.num_pixels), num_circles=NC)

# Plot the full approximation and the NC-degree 
# approximation for t from 1-R
# R = 50
# plt.plot(xFT.t_vals[0:R], xFT.fxn_vals[0:R])
# plt.plot(xFT.t_vals[0:R], xFT.fourier_approximation[0:R], "--")
# plt.plot(xFT.t_vals[0:R], xFT.circles_approximation[0:R], ".-")

# plt.plot(yFT.t_vals[0:R], yFT.fxn_vals[0:R])
# plt.plot(yFT.t_vals[0:R], yFT.fourier_approximation[0:R], "--")
# plt.plot(yFT.t_vals[0:R], yFT.circles_approximation[0:R], ".-")



# Visualize the approximation as 2D image
# plt.plot(xFT.fxn_vals, yFT.fxn_vals)
plt.plot(xFT.fourier_approximation, yFT.fourier_approximation)
# plt.plot(xFT.fourier_approximation[0], yFT.fourier_approximation[0], 'o', color='red')



# http://davidbau.com/conformal/#z

n_points = 100

N = 1





plt.show()
