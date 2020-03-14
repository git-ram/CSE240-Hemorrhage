from cachetools import TTLCache
from numbers import Number
import pydicom
class DataLoader:
    def __init__(self, base_file_path, cache_size=500, ttl_seconds=20):
        self.base_file_path = base_file_path
        self.cache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)

    ##will apply only brain windowing while loading the image for now. Need to change this to apply all windowing functions.
    def load_image(self, image_id):
        if image_id in self.cache:
            return self.cache[image_id]
        else:
            current_file = pydicom.dcmread(self.base_file_path + image_id + '.dcm')
            pixel_array = self.brain_window(current_file)
            self.cache[image_id] = pixel_array
            return pixel_array

    def trigger_expire(self):
        self.cache.expire()

    def brain_window(self, img):

        window_center = img.WindowCenter if isinstance(img.WindowCenter, Number) else img.WindowCenter[0]
        window_width = img.WindowWidth if isinstance(img.WindowWidth, Number) else img.WindowWidth[0]
        slope, intercept = img.RescaleSlope, img.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = img.pixel_array
        img = img * slope + intercept
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        # Normalize
        img = (img - img_min) / (img_max - img_min)
        return img
