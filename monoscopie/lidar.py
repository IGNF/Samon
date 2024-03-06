from osgeo import gdal
import numpy as np

class MNSLidar:
    def __init__(self, mns_path:str) -> None:
        self.mns_path:str = mns_path
        self.geotransform:tuple = None
        self.mns:np.array = None

        self.open_mns()


    def open_mns(self):
        inputds = gdal.Open(self.mns_path)
        self.geotransform = inputds.GetGeoTransform()
        self.mns = inputds.ReadAsArray()

    def get_value(self, x, y):
        x_i = int((x - self.geotransform[0]) / self.geotransform[1])
        y_i = int((y - self.geotransform[3]) / self.geotransform[5])
        if y_i >= 0 and y_i < self.mns.shape[0] and x_i >= 0 and x_i < self.mns.shape[1]:
            return self.mns[y_i, x_i]
        return None