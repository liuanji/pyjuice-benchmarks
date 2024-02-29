import torch


class RGB2YCoCg():
    def __init__(self, channel_last = False, rgb_range = (-1, 1)):
        self.channel_last = channel_last
        self.rgb_range = rgb_range

    def __call__(self, x):
        if self.channel_last:
            R, G, B = x[:,:,0], x[:,:,1], x[:,:,2]
        else:
            R, G, B = x[0,:,:], x[1,:,:], x[2,:,:]

        R = (R - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])
        G = (G - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])
        B = (B - self.rgb_range[0]) / (self.rgb_range[1] - self.rgb_range[0])

        Co  = R - B;
        tmp = B + Co/2;
        Cg  = G - tmp;
        Y   = tmp + Cg/2;

        # Make the range of Y to be [-1, 1]
        Y = Y * 2 - 1

        if self.channel_last:
            return torch.stack((Y, Co, Cg), dim = 2)
        else:
            return torch.stack((Y, Co, Cg), dim = 0)


class YCoCg2RGB():
    def __init__(self, channel_last = True):
        self.channel_last = channel_last

    def __call__(self, x):

        assert x.min() >= -1 and x.max() <= 1

        if self.channel_last:
            Y, Co, Cg = x[:,:,0], x[:,:,1], x[:,:,2]
        else:
            Y, Co, Cg = x[0,:,:], x[1,:,:], x[2,:,:]

        # Convert the range of Y back to [0, 1]
        Y = (Y + 1) / 2

        tmp = Y - Cg/2;
        G   = Cg + tmp;
        B   = tmp - Co/2;
        R   = B + Co;

        if self.channel_last:
            return torch.stack((R, G, B), dim = 2)
        else:
            return torch.stack((R, G, B), dim = 0)