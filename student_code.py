import numpy as np
import math

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.
  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)
  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """
  # checking that the filter dimension is odd
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  # store dimension of filter in the below vars
  filterHeight, filterWidth = filter.shape 

  # store dimension of image in the below vars
  imageHeight, imageWidth, channel = image.shape 
 
  # creating an array of same size as original image and  the final filtered image would be saved in this array
  filteredfinalImage = np.zeros((imageHeight, imageWidth, channel)) 
  
  # filling the number of rows to be padded in paddingRows variable
  paddingRows = (filterHeight - 1) // 2 

  # filling the number of columns to be padded in paddingColumns variable
  paddingColumns = (filterWidth - 1) // 2

  # create a padding tupple 
  applypadding = ((paddingRows, paddingRows), (paddingColumns, paddingColumns), (0, 0))  
  
  # final padded image after padding will be stored in finalpaddedImage
  finalpaddedImage = np.pad(image, applypadding, "reflect") 

  # Using convolution to apply the filter
  for m in range(imageHeight):
    for n in range(imageWidth):
      for c in range(channel):
        filteredfinalImage[m, n, c] = (filter * finalpaddedImage[m : m + filterHeight, n : n + filterWidth, c]).sum()
  
  # return filtered image
  return filteredfinalImage  

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.
  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  - cutoff_frequency
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)
  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  
  ############################
  ### TODO: YOUR CODE HERE ###

  # Filter dimension is calculated using cutoff_frquency
  cutoff_freq = 7

  # Calculating filter size
  filterSize = 4 * cutoff_freq + 1
  
  # Creating Gaussian filter 
  
  gaussFilter = np.array([np.exp(-(x-cutoff_freq) * (x-cutoff_freq) / 2*cutoff_freq*cutoff_freq) / np.sqrt(2 *np.pi*cutoff_freq*cutoff_freq) for x in range(0, filterSize)])
  
  # Filter values are converted to get the summation as 1.0
  gaussFilter = gaussFilter/(gaussFilter.sum()) 
 
  # create gaussian kernel
  gaussianFilter = np.outer(gaussFilter, gaussFilter)
 
  # Obtaining low frequency image using my_imfilter function
  low_frequencies = my_imfilter(image1, gaussianFilter)

  # Obtaining high frequency image using my_imfilter function
  high_frequencies = image2 - my_imfilter(image2, gaussianFilter)

  # Obtaining hybrid image by summation of low and high frequency image outputs
  hybrid_image = low_frequencies + high_frequencies

  # clipping values of hybrid image obtained above
  # values less than 0 would be clipped to 0 and more than 1 would be clipped to 1
  hybrid_image = np.clip(hybrid_image , 0, 1)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image

