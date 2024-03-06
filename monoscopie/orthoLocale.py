from __future__ import annotations
import numpy as np
from osgeo import gdal
import os
from scipy import ndimage
from shapely.geometry import Point, Polygon
import geopandas
import shutil
from pysocle.photogrammetry.shot import Shot
from pysocle.photogrammetry.ta import Ta
from sklearn.feature_extraction.image import extract_patches_2d
from typing import Tuple
from .tool import print_log, save_image
from .pva import Pva

class OrthoLocale:

    """
    Création d'une ortho construite à partir d'une pva
    """

    def __init__(self, resolution: float, center: Point, shot: Shot, size: int, ta: Ta, pva_path: str, projet, path_save_pi: str, path_save_dec: str, dz=0.1) -> None:
        self.resolution = resolution
        self.center = center
        self.shot:Shot = shot
        self.size = size
        self.ta:Ta = ta
        self.pva_path = pva_path

        self.pva = Pva(self.pva_path, path_save_pi)
        
        self.path_save_pi = path_save_pi
        self.path_save_dec = path_save_dec
        self.projet = projet
        self.dz = dz

        self.calcul_valide = True

        self.epip = None
        
        
        self.ortho: np.array = None
        self.i_j: Tuple[int, int] = (None, None)
        self.correlation: float = None
        self.ground_terrain: Point = None
        self.z_min: float = None
        self.z_max: float = None

        demi_size = int((self.size-1)/2)
        self.x_max = self.center.x + self.resolution * demi_size
        self.y_min = self.center.y - self.resolution * demi_size
        self.x_min = self.center.x - self.resolution * demi_size
        self.y_max = self.center.y + self.resolution * demi_size

        self.emprise = Polygon(((self.x_min, self.y_min), (self.x_min, self.y_max), (self.x_max, self.y_max), (self.x_max, self.y_min), (self.x_min, self.y_min)))



        #On crée les répertoiores "decalage" et "orthos_locales"
        if not os.path.exists(path_save_pi):
            os.makedirs(path_save_pi, exist_ok=True)

        if not os.path.exists(path_save_dec):
            os.makedirs(path_save_dec, exist_ok=True)
        

    def save(self) -> None:
        """
        On sauvegarde l'ortho crée à partir de la prise de vue
        TODO est-ce vraiment utile de la sauvegarder pour les modes pvaGeomEpip et pva ?
        """

        geotransform = (self.x_min-self.resolution/2, self.resolution, 0, self.y_max+self.resolution / 2, 0, -self.resolution)
        save_image(os.path.join(self.path_save_pi, self.shot.image+".tif"), self.ortho, geotransform=geotransform)

        shutil.copy(os.path.join(self.path_save_pi, self.shot.image+".tif"), os.path.join(self.path_save_dec, self.shot.image+".tif"))


    def compute_correlation_2(self, bd_ortho: np.array, orthos_pva: np.array, orthos_pva_shape:int, precis=False)->Tuple[int, int, float]:
        """
        Calcule la corrélation entre un extrait de bd_ortho et une pile d'orthos issues de la pva
        """
        size_bd_ortho = bd_ortho.shape[1]

        bd_ortho_tild = (bd_ortho - np.mean(bd_ortho)) / np.std(bd_ortho)
        orthos_pva_tild = (orthos_pva - np.mean(orthos_pva)) / np.std(orthos_pva)
        correlation = 1 - ( np.sum((bd_ortho_tild-orthos_pva_tild)**2, axis=(1,2)) / orthos_pva_tild.shape[1]**2)/2
        correlation_max = np.max(correlation)
        indice = np.argmax(correlation)

        denominateur = orthos_pva_shape - size_bd_ortho +1
        if precis and indice-denominateur >=0 and indice+denominateur < correlation.shape[0] and indice-1 >=0 and indice+1 < correlation.shape[0]:
            i1 = indice // denominateur + size_bd_ortho//2
            i, correlation1_max = self.solve_system(i1-1, i1, i1+1, correlation[indice-denominateur], correlation_max, correlation[indice+denominateur])
            j1 = indice % denominateur + size_bd_ortho//2
            j, correlation2_max = self.solve_system(j1-1, j1, j1+1, correlation[indice-1], correlation_max, correlation[indice+1])
            correlation_max = (correlation1_max+correlation2_max)/2
        else:
            i = indice // denominateur + size_bd_ortho//2
            j = indice % denominateur + size_bd_ortho//2
        return i, j, correlation_max


    def solve_system(self, x0, x1, x2, c0, c1, c2):
        """
        Résolution du système ax*2 + bx + c = corr avec :
            - x : coordonnées du point suivant un axe
            - corr : score de corrélation pour le point x
            - a, b et c : inconnues : paramètres de la fonction
        x et corr sont connus pour trois points : c0 = f(x0), c1 = f(x1), c2 = f(x2)
        """
        
        # Construction des matrices
        A = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]])
        B = np.array([[c0], [c1], [c2]])
        
        # résolution du système
        X = np.linalg.solve(A, B)
        a = X[0,0]
        b = X[1,0]
        c = X[2,0]

        # On récupère la coordonnée x pour laquelle la fonction f est maximale
        x_max = -b / (2*a)
        # On récupère la valeur maximale de la fonction
        correlation_max = -b**2/(4*a) + c
        return x_max, correlation_max



    
    def compute_correlation(self, bd_ortho: np.array) -> None:
        """
        Calcule la corrélation entre l'extrait de BD Ortho et l'ortho locale.
        On fait d'abord la corrélation sur un sous-échantillonnage 4;  puis on affine la corrélation à la résolution initiale.
        Passer par le sous-échantillonnage permet d'accélérer sensiblement le temps de calcul
        """

        # On fait une première recherche sur un sous-échantillonnage d'un facteur 4
        # On sous-échantillonne la bd ortho par 4
        bd_ortho_sous_ech4 = bd_ortho[0,::4,::4]
        size_bd_ortho_sous_ech4 = bd_ortho_sous_ech4.shape[1]

        #On sous-échantillonne l'ortho de la pva par 4 par 4
        ortho_pva_sous_ech4 = self.ortho[0, ::4, ::4]
        orthos_pva = extract_patches_2d(ortho_pva_sous_ech4, (size_bd_ortho_sous_ech4, size_bd_ortho_sous_ech4))
        i, j, correlation_max = self.compute_correlation_2(bd_ortho_sous_ech4, orthos_pva, ortho_pva_sous_ech4.shape[1])
        self.i_j = (i*4, j*4)
        self.correlation = correlation_max
            
        size_bd_ortho = bd_ortho.shape[1]
        i_haut_gauche = int(i*4 - (size_bd_ortho-1)/2 - 7)
        j_haut_gauche = int(j*4 - (size_bd_ortho-1)/2 - 7)
        i_bas_droit = int(i_haut_gauche + size_bd_ortho + 14)
        j_bas_droit = int(j_haut_gauche + size_bd_ortho + 14)

        if i_haut_gauche < 0 or i_bas_droit > self.ortho.shape[1] or j_haut_gauche < 0 or j_bas_droit > self.ortho.shape[2]:
            self.i_j = (0, 0)
            self.correlation = 0
        else:
            orthos_pvas_extraction = self.ortho[0,i_haut_gauche:i_bas_droit, j_haut_gauche:j_bas_droit]
            orthos_pva = extract_patches_2d(orthos_pvas_extraction, (size_bd_ortho, size_bd_ortho))
            i, j, correlation_max = self.compute_correlation_2(bd_ortho[0,:,:], orthos_pva, orthos_pvas_extraction.shape[1], precis=True)
            self.i_j = (i_haut_gauche+i, j_haut_gauche+j)
            self.correlation = correlation_max
    

    def convert_image_to_ground(self) -> None:
        """
        Convertit les coordonnées images en Lambert 93
        """

        x = self.x_min + self.resolution*(self.i_j[1])
        y = self.y_max - self.resolution*(self.i_j[0])

        # On récupère l'altitude du point
        z = self.projet.ta.project.dem.get(x, y).item()
        self.ground_terrain = Point(x, y, z)

    def save_correlation(self, methode: str) -> None:
        """
        Sauvegarde le centre de corrélation

        pi : première corrélation avec la pseudo-intersection
        epip : corrélation sur la droite épipolaire
        imp : corrélation améliorée 
        """
        d = {'col1': [self.shot.image], 'geometry': self.ground_terrain}
        gdf = geopandas.GeoDataFrame(d, crs="EPSG:2154")
        if methode == "pi":
            gdf.to_file(os.path.join(self.path_save_pi, self.shot.image+".shp"))
        elif methode == "epip":
            gdf.to_file(os.path.join(self.path_save_pi, self.shot.image+"_point_epip.shp"))
        elif methode == "imp":
            gdf.to_file(os.path.join(self.path_save_pi, self.shot.image+"_point_imp.shp"))
        else:
            gdf.to_file(os.path.join(self.path_save_dec, self.shot.image+".shp"))

    def get_sommet(self) -> np.array:
        """
        Retourne le sommet de prise de vue
        """
        image_conical = self.shot.imc
        return np.array([image_conical.x_pos, image_conical.y_pos, image_conical.z_pos.item()])

    def get_ground_terrain(self) -> np.array:
        """
        Retourne le point terrain de la corrélation
        """
        return np.array([self.ground_terrain.x, self.ground_terrain.y, self.ground_terrain.z])

    def get_nadir(self) -> np.array:
        """
        Retourne le nadir du sommet de prise de vue
        """
        return np.array([self.shot.imc.x_nadir, self.shot.imc.y_nadir, self.shot.imc.z_nadir])


    def get_droite_sous_ech(self, image_maitresse: OrthoLocale, z_min:float, z_max:float) -> np.array:
        """
        On construit un sous-échantillonnage de la droite entre le sommet de prise de vue et le point terrain de la corrélation

        Tous les points sont répartis de manière régulière sur l'axe z
        z min : z du point terrain - 20 mètres
        z max : z du point terrain + 100 mètres
        """

        P0_m = image_maitresse.get_sommet()
        P0 = self.shot.imc.system.world_to_euclidean(P0_m[0], P0_m[1], P0_m[2])
        P0 = np.array([P0[0].item(), P0[1].item(), P0[2].item()])

        # Le point récupéré n'est pas le résultat de l'intersection des corrélations,
        # mais les coordonnées du point trouvé par corrélation sur cette ortholocale
        # On utilise comme z les données du MNT, mais ce n'est pas un problème car on a utilisé ce MNT pour construire l'ortholocale
        P_g_m = image_maitresse.get_ground_terrain()
        P_g = self.shot.imc.system.world_to_euclidean(P_g_m[0], P_g_m[1], P_g_m[2])
        P_g = np.array([P_g[0].item(), P_g[1].item(), P_g[2].item()])
        
        M = P_g - P0

        if z_min and z_max:
            z_min_eucli = self.shot.imc.system.world_to_euclidean(P_g_m[0], P_g_m[1], z_min)[2].item()
            z_max_eucli = self.shot.imc.system.world_to_euclidean(P_g_m[0], P_g_m[1], z_max)[2].item()
            z = np.arange(z_min_eucli, z_max_eucli, self.dz)
        else:
            z = np.arange(P_g[2]-20, P_g[2]+100, self.dz)
        l = (z - P0[2]) / M[2]

        P0 = np.repeat(P0.reshape((1, 3)), l.shape[0], axis=0)
        M = np.repeat(M.reshape((1, 3)), l.shape[0], axis=0)
        l = np.repeat(l.reshape((-1, 1)), 3, axis=1)
        
        points = P0 + l * M
        x, y, z = self.shot.imc.system.euclidean_to_world(points[:,0], points[:,1], points[:,2])

        return np.concatenate((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))), axis=1)


    def create_small_ortho_numpy(self, x: np.array, y: np.array, size: int, canaux:int) -> np.array:
        """
        Crée une collection d'orthos à partir de la pva

        x : numpy array, coordonnées x du centre des petites orthos à produire le long de la droite épipolaire
        y : numpy array, coordonnées y du centre des petites orthos à produire le long de la droite épipolaire 
        size : int, taille des petites orthos à produire
        """
        demi_size = (size-1)/2
        nb_ortho = x.shape[0]
        x_min = x - self.resolution * demi_size
        x_max = x + self.resolution * demi_size
        x_interval = np.linspace(x_min, x_max, size)


        y_min = y - self.resolution * demi_size
        y_max = y + self.resolution * demi_size

        y_interval = np.flip(np.linspace(y_min, y_max, size), axis=0)

        xv = np.zeros((nb_ortho, size**2))
        yv = np.zeros((nb_ortho, size**2))
        for i in range(x.shape[0]):
            xv_temp, yv_temp = np.meshgrid(x_interval[:,i], y_interval[:,i])
            xv[i,:] = xv_temp.reshape((size**2))
            yv[i,:] = yv_temp.reshape((size**2))

        xv_reshaped = xv.reshape((size**2* nb_ortho))
        yv_reshaped = yv.reshape((size**2* nb_ortho))

        #On récupère l'altitude en tout point
        z = self.ta.project.dem.get(xv_reshaped, yv_reshaped)
        
        self.z_min = float(np.min(z))
        self.z_max = float(np.max(z))


        c, l = self.shot.imc.world_to_image(xv_reshaped, yv_reshaped, z)
        c = c.reshape((nb_ortho, size**2))
        l = l.reshape((nb_ortho, size**2))

        
        inputds = gdal.Open(self.pva_path)
        max_c_raster = inputds.RasterXSize
        max_l_raster = inputds.RasterYSize

        #On supprime les orthophotos qui débordent de la pva
        c, l = self.extract(c >= 0, c, l, size)
        c, l = self.extract(c < max_c_raster-1, c, l, size)
        c, l = self.extract(l >= 0, c, l, size)
        c, l = self.extract(l < max_l_raster-1, c, l, size)

        if c.shape[0] == 0:
            return None

        min_c = int(np.floor(np.min(c)))
        max_c = int(np.ceil(np.max(c)))
        min_l = int(np.floor(np.min(l)))
        max_l = int(np.ceil(np.max(l)))


        #Lecture directe de la zone
        image = inputds.ReadAsArray(min_c, min_l, max_c-min_c+1, max_l-min_l+1)

        l = l.reshape((-1, ))
        c = c.reshape((-1, ))

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

        else:
            print_log("Erreur, le nombre de canaux n'est pas correct. Il vaut {} au lieu de 1 ou 3".format(canaux))
            ortho = None

        return ortho


    def create_vignette(self, x: np.array, y: np.array, size: int, canaux:int) -> np.array:
        """
        Crée une collection d'orthos à partir de la pva

        TODO doublon de create_small_ortho_numpy

        x : numpy array, coordonnées x du centre des petites orthos à produire le long de la droite épipolaire
        y : numpy array, coordonnées y du centre des petites orthos à produire le long de la droite épipolaire 
        size : int, taille des petites orthos à produire
        """

        ortho = self.create_small_ortho_numpy(x, y, size, canaux)
        if ortho is not None:
            ortho = np.squeeze(ortho)
        return ortho
        

    def extract(self, condition: np.array, c: np.array, l: np.array, size: int) -> Tuple[np.array, np.array]:
        """
        Supprime dans c et l les données qui ne respectent pas la condition
        """
        condition_tile = np.tile(np.all(condition, axis = 1).reshape((-1, 1)), (1, size**2))        
        c = np.extract(condition_tile, c).reshape((-1, size**2))
        l = np.extract(condition_tile, l).reshape((-1, size**2))
        return c, l