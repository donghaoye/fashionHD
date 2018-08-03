'''
Derived from flownet2.0
'''

import numpy as np
import cv2


def readFlow(fn):
    """
    Derived from flownet2.0    
    """
    f = open(fn, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(fn, flow):
    """
    Derived from flownet2.0    
    """
    f = open(fn, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()


def write_flow_and_mask(fn, flow, mask):
    '''
    Save flow (float data) and mask (uint data) to one file.
    Input:
        fn: fine name
        flow: (H, W, 2), float32
        mask: (H, W), uint8
    '''
    assert flow.shape[:2]==mask.shape
    with open(fn, 'wb') as f:
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)
        mask.astype(np.uint8).tofile(f)

def read_flow_and_mask(fn):
    '''
    Recover flow and mask saved by "write_flow_and_mask"
    '''
    with open(fn, 'rb') as f:
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width*height*2).reshape((height, width, 2))
        mask = np.fromfile(f, np.uint8, width*height).reshape((height, width))
    return flow, mask


def warp_image(img, flow):
    h, w = flow.shape[:2]
    m = flow.astype(np.float32)
    m[:,:,0] += np.arange(w)
    m[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, m, None, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    return res


def flow_to_rgb(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3))
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb