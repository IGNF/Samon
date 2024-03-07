import numpy as np
from osgeo import gdal
from scipy import ndimage
from .epipolarGeometry import EpipolarGeometry
from .tool import save_image
from .shot import Shot
from typing import Tuple

class Pva:

    def __init__(self, path_pva, path_save_pi) -> None:
        self.path_pva = path_pva
        self.path_save_pi = path_save_pi

        # Si ce n'est pas renseigné avec la méthode set_geom_epip(), alors lorsque 
        # l'on créera des vignettes avec create_small_ortho_numpy(),
        # ce ne sera pas en géométrie épipolaire mais une simple extraction de la pva
        self.E:np.array = None
        self.epipolarGeometry:EpipolarGeometry = None
        self.shot:Shot = None


    def set_geom_epip(self, epipolarGeometry:EpipolarGeometry, E:np.array, shot:Shot):
        """
        Associe une matrice de passage en géométrie épipolaire
        """

        self.epipolarGeometry = epipolarGeometry
        self.E = E
        self.shot = shot

    
    def write_vignette(self, c: np.array, l: np.array, size: int, name: str) -> None:
        vignette = self.create_small_ortho_numpy(c, l, size, canaux=1)
        save_image(name, vignette)


    def create_vignette(self, c: np.array, l: np.array, size: int, canaux: int) -> np.array:

        if self.epipolarGeometry is None:
            return self.create_vignette_without_epip_geom(c, l, size, canaux)
        else:
            return self.create_small_ortho_numpy(c, l, size, canaux)






    def create_vignette_without_epip_geom(self, c: np.array, l: np.array, size: int, canaux: int) -> np.array:

        #//TODO : ce n'est pas vraiment une ortho, il faudrait changer le nom 
        """
        Crée une collection d'orthos à partir de la pva. 
        Les coordonnées c et l doivent être des entiers.
        On ne tient pas compte de la géométrie épipolaire dans cette fonction

        c : numpy array, coordonnées colonne du centre des imagettes à produire le long de la droite épipolaire
        l : numpy array, coordonnées ligne du centre des imagettes à produire le long de la droite épipolaire 
        size : int, taille des imagettes à produire en pixel
        """        
        
        demi_size = (size-1)/2
        nb_ortho = c.shape[0]
        c_min = c - demi_size
        c_max = c + demi_size
        c_interval = np.linspace(c_min, c_max, size)

        l_min = l - demi_size
        l_max = l + demi_size

        l_interval = np.linspace(l_min, l_max, size)

        cv = np.zeros((nb_ortho, size**2))
        lv = np.zeros((nb_ortho, size**2))
        for i in range(c.shape[0]):
            cv_temp, lv_temp = np.meshgrid(c_interval[:,i], l_interval[:,i])
            cv[i,:] = cv_temp.reshape((size**2))
            lv[i,:] = lv_temp.reshape((size**2))
        
        inputds = gdal.Open(self.path_pva)
        max_c_raster = inputds.RasterXSize
        max_l_raster = inputds.RasterYSize

        #On supprime les orthophotos qui débordent de la pva
        cv, lv = self.extract(cv >= 0, cv, lv, size)
        cv, lv = self.extract(cv < max_c_raster-1, cv, lv, size)
        cv, lv = self.extract(lv >= 0, cv, lv, size)
        cv, lv = self.extract(lv < max_l_raster-1, cv, lv, size)

        if cv.shape[0] == 0:
            return None

        min_c = int(np.floor(np.min(cv)))
        max_c = int(np.ceil(np.max(cv)))
        min_l = int(np.floor(np.min(lv)))
        max_l = int(np.ceil(np.max(lv)))

        #Lecture directe de la zone
        image = inputds.ReadAsArray(min_c, min_l, max_c-min_c+1, max_l-min_l+1)

        l = lv.reshape((-1, ))
        c = cv.reshape((-1, ))

        l = l.astype(int)
        c = c.astype(int)

        if canaux == 1:
            value_r = image[0,l-min_l, c-min_c]
            ortho = value_r.reshape((-1, 1, size, size))
        elif canaux == 3:
            value_r = image[0,l-min_l, c-min_c].reshape((-1, 1, size, size))
            value_g = image[1,l-min_l, c-min_c].reshape((-1, 1, size, size))
            value_b = image[2,l-min_l, c-min_c].reshape((-1, 1, size, size))
            ortho = np.concatenate((value_r, value_g, value_b), axis=1)
            self.max_l = max_l
            self.max_c = max_c
            self.min_l = min_l
            self.min_c = min_c
        return np.squeeze(ortho)




    def create_small_ortho_numpy(self, c: np.array, l: np.array, size: int, canaux: int) -> np.array:

        #//TODO : ce n'est pas vraiment une ortho, il faudrait changer le nom 
        """
        Crée une collection d'orthos à partir de la pva

        c : numpy array, coordonnées colonne du centre des imagettes à produire le long de la droite épipolaire
        l : numpy array, coordonnées ligne du centre des imagettes à produire le long de la droite épipolaire 
        size : int, taille des imagettes à produire en pixel
        """        
        
        demi_size = (size-1)/2
        nb_ortho = c.shape[0]
        c_min = c - demi_size
        c_max = c + demi_size
        c_interval = np.linspace(c_min, c_max, size)

        l_min = l - demi_size
        l_max = l + demi_size

        l_interval = np.linspace(l_min, l_max, size)

        cv = np.zeros((nb_ortho, size**2))
        lv = np.zeros((nb_ortho, size**2))
        for i in range(c.shape[0]):
            cv_temp, lv_temp = np.meshgrid(c_interval[:,i], l_interval[:,i])
            cv[i,:] = cv_temp.reshape((size**2))
            lv[i,:] = lv_temp.reshape((size**2))

        cv = cv.reshape((-1))
        lv = lv.reshape((-1))

        # Pour chaque pixel de la grille, on récupère les coordonnées dans l'image
        if self.epipolarGeometry is not None:
            cv, lv = self.epipolarGeometry.epip_to_image(cv, lv, self.shot, self.E, use_dh=False)

        cv = cv.reshape((nb_ortho, size**2))
        lv = lv.reshape((nb_ortho, size**2))
        
        inputds = gdal.Open(self.path_pva)
        max_c_raster = inputds.RasterXSize
        max_l_raster = inputds.RasterYSize

        #On supprime les orthophotos qui débordent de la pva
        cv, lv = self.extract(cv >= 0, cv, lv, size)
        cv, lv = self.extract(cv < max_c_raster-1, cv, lv, size)
        cv, lv = self.extract(lv >= 0, cv, lv, size)
        cv, lv = self.extract(lv < max_l_raster-1, cv, lv, size)

        if cv.shape[0] == 0:
            return None

        min_c = int(np.floor(np.min(cv)))
        max_c = int(np.ceil(np.max(cv)))
        min_l = int(np.floor(np.min(lv)))
        max_l = int(np.ceil(np.max(lv)))


        #Lecture directe de la zone
        image = inputds.ReadAsArray(min_c, min_l, max_c-min_c+1, max_l-min_l+1)

        l = lv.reshape((-1, ))
        c = cv.reshape((-1, ))

        if canaux == 1:
            
            value_r = ndimage.map_coordinates(image[0,:,:], np.vstack([l-min_l, c-min_c]))
            ortho = value_r.reshape((-1, 1, size, size))
        elif canaux == 3:
            value_r = ndimage.map_coordinates(image[0,:,:], np.vstack([l-min_l, c-min_c])).reshape((-1, 1, size, size))
            value_g = ndimage.map_coordinates(image[1,:,:], np.vstack([l-min_l, c-min_c])).reshape((-1, 1, size, size))
            value_b = ndimage.map_coordinates(image[2,:,:], np.vstack([l-min_l, c-min_c])).reshape((-1, 1, size, size))
            ortho = np.concatenate((value_r, value_g, value_b), axis=1)
            self.max_l = max_l
            self.max_c = max_c
            self.min_l = min_l
            self.min_c = min_c
        return np.squeeze(ortho)


    def extract(self, condition: np.array, c: np.array, l: np.array, size: int) -> Tuple[np.array, np.array]:
        """
        Supprime dans c et l les données qui ne respectent pas la condition
        """
        condition_tile = np.tile(np.all(condition, axis = 1).reshape((-1, 1)), (1, size**2))        
        c = np.extract(condition_tile, c).reshape((-1, size**2))
        l = np.extract(condition_tile, l).reshape((-1, size**2))
        return c, l