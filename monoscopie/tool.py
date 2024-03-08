import os
from osgeo import gdal, osr
import numpy as np
from qgis.core import QgsMessageLog, Qgis

def print_log(message:str)->None:
    
    print(message)
    #QgsMessageLog.logMessage(message, tag="Infos Samon", level=Qgis.MessageLevel.Info)
    



def save_image(name:str, image:np.array, dtype=gdal.GDT_Byte, geotransform=None):

    if image is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(2154)

        driver = gdal.GetDriverByName('GTiff')
        if image.ndim == 2:
            outRaster = driver.Create(name, image.shape[1], image.shape[0], 1, dtype)
            if geotransform is not None:
                outRaster.SetGeoTransform(geotransform)
                outRaster.SetProjection(outSpatialRef.ExportToWkt())
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(image)
        else:
            outRaster = driver.Create(name, image.shape[2], image.shape[1], image.shape[0], dtype)
            if geotransform is not None:
                outRaster.SetGeoTransform(geotransform)
                outRaster.SetProjection(outSpatialRef.ExportToWkt())
            for i in range(image.shape[0]):
                outband = outRaster.GetRasterBand(i+1)
                outband.WriteArray(image[i, :, :])
        outRaster = None