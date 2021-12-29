import numpy as np


class Reverse_scale(object):
    """rescale the image input data to output_rescale image
    parameters:
        output_size : shows the reverse scale image output"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img_shape, pose):
        """
        this args for using when img scale with (mean , std)
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        """
        # len_pose = len(pose)
        origin_img_left, origin_img_top, origin_img_width ,origin_img_height = img_shape[0], img_shape[1], img_shape[2], img_shape[3]

        w_ratio = float(origin_img_width)/float(self.output_size[0])
        h_ratio = float(origin_img_height) /float(self.output_size[1])
        i_scale = max(w_ratio, h_ratio)

        if h_ratio > w_ratio:
            w_diff = self.output_size[0]/2-origin_img_width/(2*i_scale)
            pose[...,0]-=w_diff
        else:
            h_diff = self.output_size[1]/2-origin_img_height/(2*i_scale)
            pose[...,1]-=h_diff
        
        pose*=i_scale
        pose[...,0]+=origin_img_left
        pose[...,1]+=origin_img_top
        