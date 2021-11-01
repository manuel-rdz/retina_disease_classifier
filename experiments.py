import cv2
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter


# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

dim = (500, 500)
bright_inc = 80

def show_img(title, img):
    n_img = cv2.resize(img, dim)
    n_img = np.where((255 - n_img) < bright_inc, 255,n_img+bright_inc)
    #(thresh, im_bw) = cv2.threshold(n_img, 15, 255, cv2.THRESH_BINARY)
    cv2.imshow(title, n_img)

img_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop\\im0208.png'
#'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training\\471.png'



img = cv2.imread(img_path)[:, :, 1]
#cv2.imshow('green_channel', img)

gauss_img = gaussian_filter(img, 0.463)
show_img('gauss img', gauss_img)


op_size = 8
cl_size = 16

# Normal top-hat
inv_img = cv2.bitwise_not(gauss_img)
cv2.imshow('complement img', cv2.resize(inv_img, dim))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op_size, op_size))
top_hat_img = cv2.morphologyEx(inv_img, cv2.MORPH_TOPHAT, kernel)
#cv2.imshow('top-hat img', cv2.resize(top_hat_img, dim))

# optimized top-hat
inv_img = top_hat_img
kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op_size, op_size))
kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_size, cl_size))

opt_top_hat_img = cv2.morphologyEx(inv_img, cv2.MORPH_TOPHAT, kernel_op)
show_img('top-hat', opt_top_hat_img)
opt_top_hat_img = cv2.morphologyEx(opt_top_hat_img, cv2.MORPH_CLOSE, kernel_cl)
show_img('opt top-hat img', opt_top_hat_img)


# Homomorphic filter
homo_filter = HomomorphicFilter(a = 0.5, b = 1.5)
img_filtered = homo_filter.filter(I=opt_top_hat_img, filter_params=[2], filter='gaussian')
show_img('homomorphic filter', img_filtered)

# median filter
img_filtered = median_filter(img_filtered, (2, 2))
show_img('median filter', img_filtered)


op_size = op_size*4
cl_size = cl_size*5
# Normal top-hat
inv_img = img_filtered
#cv2.imshow('complement img', cv2.resize(inv_img, dim))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op_size, op_size))
top_hat_img = cv2.morphologyEx(inv_img, cv2.MORPH_TOPHAT, kernel)
#cv2.imshow('top-hat img', cv2.resize(top_hat_img, dim))

# optimized top-hat
inv_img2 = top_hat_img
kernel_op2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op_size, op_size))
kernel_cl2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_size, cl_size))

opt_top_hat_img2 = cv2.morphologyEx(inv_img2, cv2.MORPH_TOPHAT, kernel_op)
show_img('top-hat', opt_top_hat_img)
opt_top_hat_img = cv2.morphologyEx(opt_top_hat_img2, cv2.MORPH_CLOSE, kernel_cl)
show_img('opt top-hat img2', opt_top_hat_img2)

# optimized top-hat
#inv_img2 = img_filtered
#kernel_op2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32*2, 32*2))
#kernel_cl2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (86*2, 86*2))

#opt_top_hat_img2 = cv2.morphologyEx(inv_img2, cv2.MORPH_TOPHAT, kernel_op2)
#opt_top_hat_img2 = cv2.morphologyEx(opt_top_hat_img2, cv2.MORPH_CLOSE, kernel_cl2)
#show_img('2nd opt top-hat img', opt_top_hat_img2)


cv2.waitKey(0)
