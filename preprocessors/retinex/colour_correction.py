import cv2
import numpy as np

def get_ksize(sigma):
    ksize = int(((sigma - 0.8)/0.15) + 2.0)
    return ksize if ksize % 2 == 1 else ksize + 1

def get_gaussian_blur(img, ksize=0, sigma=5):
    if ksize == 0:
        ksize = get_ksize(sigma)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def single_scale_retinex(img, sigma=5):
    img = img.astype(np.float32) + 1.0
    blur = get_gaussian_blur(img, sigma=sigma)
    retinex = np.log1p(img) - np.log1p(blur + 1.0)
    return retinex

def msr(img, sigma_scales=[15, 81, 251]):
    msr = np.zeros(img.shape)
    for sigma in sigma_scales:
        msr += single_scale_retinex(img, sigma)
    msr = msr / len(sigma_scales)
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return msr

def color_balance(img, low_per, high_per):
    tot_pix = img.shape[1] * img.shape[0]
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100
    ch_list = [img] if len(img.shape) == 2 else cv2.split(img)
    cs_img = []
    for ch in ch_list:
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            cs_img.append(ch)
            continue
        lut = np.array([0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255)) for i in np.arange(0, 256)], dtype='uint8')
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)
    return np.squeeze(cs_img) if len(cs_img) == 1 else cv2.merge(cs_img)

def msrcr(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    img = img.astype(np.float64) + 1.0
    msr_img = msr(img, sigma_scales)
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    msrcr = G * (msr_img * crf - b)
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    msrcr = color_balance(msrcr, low_per, high_per)
    return msrcr

def msrcp(img, sigma_scales=[15, 80, 250], low_per=1, high_per=1):
    int_img = (np.sum(img, axis=2) / img.shape[2]) + 1.0
    msr_int = msr(int_img, sigma_scales)
    msr_cb = color_balance(msr_int, low_per, high_per)
    B = 256.0 / (np.max(img, axis=2) + 1.0)
    BB = np.array([B, msr_cb / int_img])
    A = np.min(BB, axis=0)
    msrcp = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)
    return msrcp.astype(np.uint8)

if __name__ == "__main__":
    img_path = r'C:\Users\DuenoHfao\Desktop\Work\SCVU\mlgp_full\data\img_dataset\Car\2015_03014.png'
    img = cv2.imread(img_path)
    msrcp_img = msrcp(img, sigma_scales=[15, 80, 250], low_per=1, high_per=1)
    cv2.imshow('MSRCP', msrcp_img)
    cv2.waitKey(0)
