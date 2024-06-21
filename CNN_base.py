
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(0)
import keras
import tensorflow as tf 

# Load and preprocess the image
# Step 1: Load the image from file
img = cv2.imread("path_image") 
# Step 2: Resize the image to 400x400 pixels
img = cv2.resize(img, (400, 400))
# Step 3: Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the Conv2d class for convolution operation
class Conv2d: 
    def __init__(self, input, num_of_kernel=8, kernel_size=3, padding=1, stride=1): 
        # Step 4: Add padding to the input image
        self.input = np.pad(input, ((padding, padding), (padding, padding)), 'constant')
        self.kernel_size = kernel_size
        self.stride = stride
        # Step 5: Initialize the kernels randomly
        self.kernel = np.random.randn(num_of_kernel, self.kernel_size, self.kernel_size)
        # Step 6: Initialize the result matrix for convolution
        self.result = np.zeros(((self.input.shape[0] - self.kernel.shape[1]) // self.stride + 1,
                                (self.input.shape[1] - self.kernel.shape[2]) // self.stride + 1,
                                self.kernel.shape[0]))
    
    # Generate regions of interest (ROI) for convolution
    def getROI(self): 
        # Step 7: Yield ROI for each possible position
        for row in range((self.input.shape[0] - self.kernel.shape[1]) // self.stride + 1): 
            for col in range((self.input.shape[1] - self.kernel.shape[2]) // self.stride + 1):
                roi = self.input[row * self.stride : row * self.stride + self.kernel.shape[1],
                                 col * self.stride : col * self.stride + self.kernel.shape[2]]
                yield row, col, roi 
    
    # Perform the convolution operation
    def operate(self): 
        # Step 8: Apply each kernel to the input image and store the results
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.result[row, col, layer] = np.sum(roi * self.kernel[layer])
        return self.result

# Define the Relu class for ReLU activation
class Relu: 
    def __init__(self, input): 
        self.input = input
        # Step 9: Initialize the result matrix for ReLU activation
        self.result = np.zeros((self.input.shape[0],
                                self.input.shape[1],
                                self.input.shape[2]))
    
    # Perform the ReLU activation operation
    def operator(self): 
        # Step 10: Apply ReLU to each element in the input
        for layer in range(self.input.shape[2]): 
            for row in range(self.input.shape[0]): 
                for col in range(self.input.shape[1]): 
                    self.result[row, col, layer] = 0.1 * self.input[row, col, layer] if self.input[row, col, layer] < 0 else self.input[row, col, layer]
        return self.result

# Define the MaxPooling class for max pooling operation
class MaxPooling:
    def __init__(self, input, pooling_size):
        self.pooling_size = pooling_size 
        self.input = input
        # Step 11: Initialize the result matrix for max pooling
        self.result = np.zeros((self.input.shape[0] // self.pooling_size,
                                self.input.shape[1] // self.pooling_size,
                                self.input.shape[2]))
    
    # Perform the max pooling operation
    def operate(self): 
        # Step 12: Apply max pooling to each region of the input
        for layer in range(self.input.shape[2]): 
            for row in range(self.input.shape[0] // self.pooling_size):
                for col in range(self.input.shape[1] // self.pooling_size): 
                    self.result[row, col, layer] = np.max(self.input[row * self.pooling_size : (row + 1) * self.pooling_size,
                                                                      col * self.pooling_size : (col + 1) * self.pooling_size,
                                                                      layer])
        return self.result

# Define the Softmax class for softmax activation
class Softmax: 
    def __init__(self, input, nodes):
        self.input = input 
        self.nodes = nodes
        # Step 13: Flatten the input matrix
        self.flatten = self.input.flatten()
        print(self.flatten.shape)
        # Step 14: Initialize weights and biases randomly
        self.weight = np.random.randn(self.flatten.shape[0]) / self.flatten.shape[0]
        self.bias = np.random.randn(self.nodes)
    
    # Perform the softmax computation
    def compute(self): 
        # Step 15: Compute the weighted sum and apply softmax
        totals = np.dot(self.flatten, self.weight) + self.bias
        exp = np.exp(totals)
        return exp / sum(exp)

# Main function to demonstrate the operations
if __name__ == '__main__': 
    # Step 16: Perform convolution operation
    img_gray_cvt = Conv2d(img_gray, 16).operate()
    # Step 17: Perform ReLU activation
    img_gray_cvt_relu = Relu(img_gray_cvt).operator()
    # Step 18: Perform MaxPooling operation
    img_gray_cvt_relu_MaxPooling = MaxPooling(img_gray_cvt_relu, 2).operate()
    # Step 19: Perform softmax activation
    s_max = Softmax(img_gray_cvt_relu_MaxPooling, 10).compute()
    print(s_max)

    # Visualization (commented out for now)
    # for i in range(16): 
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(img_gray_cvt_relu_MaxPooling[:, :, i], cmap='gray')
    # plt.savefig('img_gray_conv2d_relu.jpg')
    # plt.show()