
class Parameters:
    

    def __init__(self, e_b_k = (3,3), e_b_r = 1, e_c_t1 =50, e_c_t2 = 200, b_h_mi = (0,110,79), b_h_ma = (24,255,255), b_b_k = (7,7) ):
        self.edge_blur_kernel = e_b_k
        self.edge_blur_rounds = e_b_r
        self.edge_canny_thresh1 = e_c_t1
        self.edge_canny_thresh2 = e_c_t2

        self.blob_hsv_min = b_h_mi
        self.blob_hsv_max = b_h_ma
        self.blob_blur_kernel = b_b_k
        self.msssim_weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) 
