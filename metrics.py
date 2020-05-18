import matplotlib.pyplot as plt

class Metric:

    def __init__(self):
        self.PSNR = []
        self.MSE = []
        self.SSIM = []
        self.MSSSIM = []
        self.EDGEMSE = []
        self.EDGERESPONSE = []
        self.BLOBRATIO = []
        self.OFFSETX = []
        self.OFFSETY = []
        self.SIZERATIO = []


        
 
    def printValues(self):
        print(self.__dict__)

    def showValues(self):
        vals = self.__dict__

        for key in vals:
            plt.plot(vals[key])
            plt.ylabel(key)
            plt.show()

