import numpy as np
from osgeo import gdal
import os
import scipy
from shapely.geometry import Point
from .tool import print_log, save_image

class Ortho:

    """def __init__(self, centre: Point, resolution: float, size: int, path_ortho: str, path_save: str):
        self.centre = centre
        self.resolution = resolution
        self.size = size
        self.path_ortho = path_ortho
        self.path_save = path_save

        if not os.path.exists(path_save):
            os.mkdir(path_save)"""


    def __init__(self, resolution: float, size: int, path_save: str, centre:Point=None, image:np.array=None, path_ortho: str=None):
        self.resolution = resolution
        self.size = size
        self.path_save = path_save
        self.centre = centre

        if not os.path.exists(path_save):
            os.mkdir(path_save)

        
        if centre is None and image is None:
            print_log("Erreur dans la construction de l'ortho")

        if centre is not None:
            
            self.path_ortho = path_ortho

        if image is not None :
            self.bd_ortho = image



    def create_ortho(self) -> None:
        """
        Crée, à partir de la BD Ortho, l'ortho locale centrée sur self.centre
        """

        print_log("self.path_ortho : {}".format(self.path_ortho))
        inputds = gdal.Open(self.path_ortho) 

        
        demi_size = (self.size-1)/2
        # On définit les coordonnées de l'angle supérieur gauche
        x_min = self.centre.x - self.resolution*demi_size
        y_max = self.centre.y + self.resolution*demi_size

        # On récupère les coordonnées de l'angle inférieur droit
        x_max = self.centre.x + self.resolution*demi_size
        y_min = self.centre.y - self.resolution*demi_size

        # On construit les couples de coordonnées pour lesquels il va falloir retrouver la radiométrie sur l'ortho initiale
        x = np.linspace(x_min, x_max, self.size) - self.resolution / 2 #Soustraire self.resolution / 2 pour tenir compte de l'origine des coordonnées pour chaque pixel
        y = np.flip(np.linspace(y_min, y_max, self.size)) + self.resolution / 2
        xv, yv = np.meshgrid(x, y)
        xv_reshaped = xv.reshape((self.size**2))
        yv_reshaped = yv.reshape((self.size**2))

        # On récupère le géoréférencement de la bd ortho
        geotransform = inputds.GetGeoTransform()
        c = (xv_reshaped - geotransform[0]) / geotransform[1]
        l = (yv_reshaped - geotransform[3]) / geotransform[5]

        # On détermine la partie de la bd ortho qui est utile pour ne pas avoir à charger toute la bd ortho
        min_c = int(np.floor(np.min(c)))
        max_c = int(np.ceil(np.max(c)))
        min_l = int(np.floor(np.min(l)))
        max_l = int(np.ceil(np.max(l)))

        # On récupère la partie utile de l'ortho
        bd_ortho = inputds.ReadAsArray(min_c, min_l, max_c-min_c+1, max_l-min_l+1, band_list=[1])

        # On construit la bd ortho locale centrée sur le point défini par l'opérateur
        l = l-min_l
        c = c-min_c
        value_r = scipy.ndimage.map_coordinates(bd_ortho, np.vstack([l, c]))
        self.bd_ortho = value_r.reshape((1, self.size, self.size))


    def get_x_min(self) -> float:
        """
        Renvoie la coordonnée x de l'angle nord-ouest
        """
        demi_size = (self.size-1)/2
        return self.centre.x - self.resolution*demi_size


    def get_y_max(self) -> float:
        """
        Renvoie la coordonnée y de l'angle nord-ouest
        """
        demi_size = (self.size-1)/2
        return self.centre.y + self.resolution*demi_size


    def save_ortho(self, nom:str) -> None:
        """
        Sauvegarde l'ortho dans "bd_ortho"
        """
        geotransform = (self.get_x_min()-self.resolution/2, self.resolution, 0, self.get_y_max()+self.resolution/2, 0, -self.resolution)
        save_image(os.path.join(self.path_save, nom), self.bd_ortho, geotransform=geotransform)

