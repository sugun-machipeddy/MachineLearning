import numpy as np
#from sklearn.cluster import KMeans
from kmeans import Kmeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour

        """

        H, W, _ = img.shape
        #print(img.shape)
        z = []
        #print(z)
        #print (img[0])

        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                z.append(img[h][w])

        #print(z)

        #print(len(img))
        #model = KMeans(n_clusters=2**self.b, n_init=3)
        model = Kmeans(k = 2**self.b)
        m = model.fit(np.array(z))
        y = model.predict(np.array(z))
        print('y =', y)
        print('means =', m)




        # TODO: fill in code here
        #raise NotImplementedError()

        #return quantized_img

    def dequantize(self, quantized_img):
        H, W = quantized_img.shape
        img = np.zeros((H,W,3), dtype='uint8')

        # TODO: fill in the values of `img` here
        raise NotImplementedError()

        return img


