import os
import shutil
from shapely import contains
import numpy as np
from osgeo import gdal
from .ortho import Ortho
from .orthoLocale import OrthoLocale
from typing import List, Tuple
from shapely.geometry import LineString, Point
import geopandas
from lxml import etree
from scipy.interpolate import griddata
from plyfile import PlyData
from pysocle.photogrammetry.shot import Shot
from .tool import print_log, save_image
from .correlation_engine import CorrelationEngine
from .epipolarGeometry import EpipolarGeometry

class Chantier:
    """
    A class for getting x, y, z coordinates from one point on BD Ortho
    """

    def __init__(self, point: Point, id: int, resolution: float, projet, type_correlation="pvaGeomEpip", sauvegarde=False):
        self.point = point
        self.id = id
        self.resolution = resolution
        self.projet = projet
        self.path = os.path.join(self.projet.resultats, "point_{}".format(self.id))
        self.sauvegarde = sauvegarde

        self.ortho_locales:List[OrthoLocale] = []
        self.ortho_locales_valides_dec:List[OrthoLocale] = []
        self.image_maitresse:OrthoLocale = None
        self.x_chap = None
        self.x_chap_dec = None
        self.x_chap_micmac = None

        self.type_correlation = type_correlation

        print_log("\n\n\nTraitement du point {}".format(id))
        print_log(self.point)
        self.create_directory()


    def create_directory(self) -> None:
        """
        Crée le répertoire contenant toutes les informations relatives à ce chantier
        """
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)


    def get_pvas(self, shots: List[Shot]) -> None:
        """
        Récupère les pvas dont l'emprise contient le point
        """
        self.pvas = []
        for shot in shots:
            if contains(shot._extent, self.point):
                self.pvas.append(shot)
        print_log("Nombre de pvas : {}".format(len(self.pvas)))

    
    def create_bd_ortho(self, image_maitresse=None) -> None:
        """
        Crée les deux orthos extraites de la BD Ortho : une grande et une petite
        """
        if image_maitresse:
            orthoLocale = OrthoLocale(self.resolution, self.point, image_maitresse, self.projet.size_bd_ortho, self.projet.ta, os.path.join(self.projet.pva, image_maitresse.image+".tif"), self.projet, os.path.join(self.path, "orthos_locales"), os.path.join(self.path, "decalage"))
            ortho = orthoLocale.create_small_ortho_numpy(np.array([orthoLocale.center.x]), np.array([orthoLocale.center.y]), orthoLocale.size, 3)
            self.bd_ortho = Ortho(orthoLocale.resolution, self.projet.size_bd_ortho, os.path.join(self.path, "bd_ortho"), centre=self.point, image=ortho[0,:,:,:])
        else:
            self.bd_ortho = Ortho(self.resolution, self.projet.size_bd_ortho, os.path.join(self.path, "bd_ortho"), centre=self.point, path_ortho=self.projet.ortho)
            self.bd_ortho.create_ortho()
        
        self.bd_ortho.save_ortho("grande_ortho.tif")


    def create_small_ortho(self):
        #On construit la petite ortho en la découpant de la première ortho, cela fait gagner du temps plutôt que de refaire une interpolation
        self.small_bd_ortho = Ortho(self.resolution, self.projet.size_small_bd_ortho, os.path.join(self.path, "bd_ortho"), centre=self.point, path_ortho=self.projet.ortho)
        dec = int((self.projet.size_bd_ortho - self.projet.size_small_bd_ortho)/2)
        small_ortho = self.bd_ortho.bd_ortho[:, dec:dec+self.projet.size_small_bd_ortho, dec:dec+self.projet.size_small_bd_ortho]
        self.small_bd_ortho.bd_ortho = small_ortho
        self.small_bd_ortho.save_ortho("petite_ortho.tif")


    def create_orthos_locales(self) -> None:
        """
        Construit pour chaque pva une ortho locale
        """
        
        # On parcourt chaque pva
        for pva in self.pvas:
            # On crée une ortho locale à partir de cette pva
            orthoLocale = OrthoLocale(self.resolution, self.point, pva, self.projet.size_orthoLocale, self.projet.ta, os.path.join(self.projet.pva, pva.image+".tif"), self.projet, os.path.join(self.path, "orthos_locales"), os.path.join(self.path, "decalage"))
            ortho = None
            if contains(pva._extent, orthoLocale.emprise):
                ortho = orthoLocale.create_small_ortho_numpy(np.array([orthoLocale.center.x]), np.array([orthoLocale.center.y]), orthoLocale.size, 3)
            # Il arrive que pour des pvas, on ne puisse pas produire d'ortho locale car le point se situe trop en extrémité d'image
            if ortho is not None:
                orthoLocale.ortho = ortho[0,:,:,:]
                orthoLocale.save()
                self.ortho_locales.append(orthoLocale)
            else:
                print_log("PVA ne permettant pas de faire une ortho locale : {}".format(orthoLocale.shot.image))
        print_log("Nombre de pvas permettant de faire une ortho complète : {}\n".format(len(self.ortho_locales)))


    def compute_correlations(self, bd_ortho: Ortho, methode: str) -> None:
        """
        Recherche pour chaque PVA le point de corrélation entre l'extrait de BD Ortho et l'ortho locale
        """

        for orthoLocale in self.ortho_locales:
            orthoLocale.compute_correlation(bd_ortho.bd_ortho)
            orthoLocale.convert_image_to_ground()
            if self.sauvegarde:
                orthoLocale.save_correlation(methode)
            print_log("Image : {}, Corrélation : {}, (x,y) : ({}, {})".format(orthoLocale.shot.image, orthoLocale.correlation, orthoLocale.get_ground_terrain()[0], orthoLocale.get_ground_terrain()[1]))


    def improve_correlations(self) -> None:
        """
        On améliore la précision de la corrélation sur les images ayant un score de corrélation supérieur à 0.9.
        Il est déjà arrivé qu'à l'issue de la première recherche (compute_correlations), 
        l'image avec le meilleur score de corrélation ne soit pas l'image maîtresse
        """

        print_log("On améliore la précision du pointé")
        """for orthoLocale in self.ortho_locales:
            if orthoLocale.correlation > 0.9:
                
                #On détermine les coordonnées autour du point de corrélation trouvé précédemment où il faut affiner la recherche
                gt = orthoLocale.get_ground_terrain()
                
                x = np.arange(gt[0] - self.resolution, gt[0] + self.resolution, self.resolution/5)
                y = np.flip(np.arange(gt[1] - self.resolution, gt[1] + self.resolution, self.resolution/5))

                xv, yv = np.meshgrid(x, y)

                xv_reshaped = xv.reshape((-1, 1))
                yv_reshaped = yv.reshape((-1, 1))

                #On construit des orthos locales centrées sur ces coordonnées
                orthos = orthoLocale.create_small_ortho_numpy(xv_reshaped, yv_reshaped, self.projet.size_bd_ortho, 1)
                orthos = np.squeeze(orthos)
                
                #On normalize les images
                orthos_tild = (orthos - np.mean(orthos, axis=(1,2)).reshape((-1, 1, 1))) / np.std(orthos, axis=(1,2)).reshape((-1, 1, 1))
                        
                bd_ortho = np.tile(self.bd_ortho.bd_ortho[0,:,:], (orthos.shape[0], 1, 1))
                bd_ortho_tild = (bd_ortho - np.mean(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))) / np.std(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))
                        
                #On calcule la corrélation
                correlation = 1 - ( np.sum((bd_ortho_tild-orthos_tild)**2, axis=(1,2)) / orthos_tild.shape[1]**2)/2

                correlation_max = np.max(correlation)
                indice = np.argmax(correlation)
                        
                x_chap = xv_reshaped[indice].item()
                y_chap = yv_reshaped[indice].item()

                #On peut sauvegarder la vignette qui corrèle le mieux
                if self.projet.log:
                    driver = gdal.GetDriverByName('GTiff')
                    outRaster = driver.Create(os.path.join(orthoLocale.path_save_pi, orthoLocale.shot.image+"_save_imp.tif"), orthos.shape[1], orthos.shape[1], 1, gdal.GDT_Byte)
                    demi_size = (self.projet.size_bd_ortho-1)/2
                    outRaster.SetGeoTransform((x_chap-orthoLocale.resolution * demi_size-orthoLocale.resolution/2, orthoLocale.resolution, 0, y_chap+ orthoLocale.resolution * demi_size+orthoLocale.resolution/2, 0, -orthoLocale.resolution))
                    outband = outRaster.GetRasterBand(1)
                    outband.WriteArray(orthos[indice, :, :])
                    outRaster = None

                orthoLocale.correlation = correlation_max
                print_log("Image : {}, Corrélation : {}, (x,y) : ({}, {})".format(orthoLocale.shot.image, orthoLocale.correlation, x_chap, y_chap)) 
                z_chap = self.projet.ta.project.dem.get(x_chap, y_chap).item()
                orthoLocale.ground_terrain = Point(x_chap, y_chap, z_chap)

                # On sauvegarde le point où la corrélation a été trouvée
                orthoLocale.save_correlation("imp")"""

        #On trie les ortho locales de façon à ce que la première soit l'image maîtresse
        self.ortho_locales = sorted(self.ortho_locales, key=lambda d: d.correlation)
        self.ortho_locales.reverse()
        self.image_maitresse = self.ortho_locales[0]


    def filter_ortho_locales(self, seuil: float, liste_meme_bande:List[str])->None:
        """
        On retire du calcul les pvas pour lesquelles la valeur de corrélation est inférieure à un seuil
        """
        for ortho_locale in self.ortho_locales:
            ajout = False
            if ortho_locale.correlation > seuil:
                if len(liste_meme_bande) > 0:
                    if ortho_locale.shot.image in liste_meme_bande:
                        ajout = True
                else:
                    ajout = True
            ortho_locale.calcul_valide = ajout

    
    def get_nb_orthos_valides(self)->None:
        """
        On compte le nombre d'orthos qui sont considérées valide pour le calcul
        """
        compte = 0
        for orthoLocale in self.ortho_locales:
            if orthoLocale.calcul_valide:
                compte += 1
        return compte


    def compute_pseudo_intersection(self) -> Tuple[np.array, int, float]:
        """
        On calcule la pseudo-intersection entre les droites : sommet de prise de vue / corrélation
        """
        # On calcule la pseudo-intersection
        residus = self.resolution + 1
        # Tant que les résidus sont supérieurs à la résolution de l'image et que l'on a encore au moins deux pvas
        while residus > self.resolution and self.get_nb_orthos_valides() > 1:
            residus, x, y, z = self.compute_least_squares()

            # Si le résidu est supérieur à la résolution, alors on supprime la pva avec le moins bon score de corrélation
            if residus > self.resolution:
                
                self.ortho_locales.reverse()
                for ortho_locale in self.ortho_locales:
                    if ortho_locale.calcul_valide:
                        ortho_locale.calcul_valide = False
                        print_log("On retire l'image {}".format(ortho_locale.shot.image))
                        print_log("")
                        print_log("On recommence le calcul")
                        break
                self.ortho_locales.reverse()

        # Si l'on n'a plus qu'une seule droite, alors on ne peut plus calculer
        if self.get_nb_orthos_valides() <= 1:
            print_log("Impossible d'avoir assez de précision pour ce point")
            return np.array([0, 0, 0]), 0, 0
        return np.array([x, y, z]), self.get_nb_orthos_valides(), residus


    def save_epip(self, ortho_locale:OrthoLocale, c_image:np.array, l_image:np.array, x_world:np.array, y_world:np.array) -> None:
        """
        On sauvegarde les épipolaires dans des fichiers
        """
        #On sauvegarde la droite épipolaire sur la pva
        s = geopandas.GeoSeries([LineString([
            (c_image[0], -l_image[0]), 
            (c_image[-1], -l_image[-1])])
        ],)
        s.to_file(os.path.join(self.path, "orthos_locales", "{}_epip_pva.shp".format(ortho_locale.shot.image)))
        
        #On sauvegarde la droite épipolaire pour contrôle
        s = geopandas.GeoSeries([LineString([
            (x_world[0], y_world[0]), 
            (x_world[-1], y_world[-1])])
        ],)
        s.to_file(os.path.join(self.path, "orthos_locales", "{}_epip.shp".format(ortho_locale.shot.image)))


    def correlation_pvaGeomEpip(self, ortho_locale, c_image, l_image):
        # On crée l'objet EpipolarGeometry qui calcule immédiatement les matrices pour passer de géométrie image à géométrie épipolaire
        epipGeom = EpipolarGeometry(self.image_maitresse.shot, ortho_locale.shot, self.projet.ta.project.dem, self.projet.pva)

        # On construit la vignette maitresse à partir de la géométrie épipolaire de l'image maitresse 
        z = self.projet.ta.project.dem.get(int(self.small_bd_ortho.centre.x), int(self.small_bd_ortho.centre.y))
        centre_pva = self.image_maitresse.shot.imc.world_to_image(self.small_bd_ortho.centre.x, self.small_bd_ortho.centre.y, z)
        c_centre_epip, l_centre_epip = epipGeom.image_to_epip(np.array([[centre_pva[0]]]), np.array([[centre_pva[1]]]), epipGeom.image1, epipGeom.r1e, use_dh=True)

        # On calcule dh au niveau du centre de la vignette maitresse
        epipGeom.calcul_dh(centre_pva)

        self.image_maitresse.pva.set_geom_epip(epipGeom, epipGeom.r1e, epipGeom.image1)
        image_maitresse_epip = self.image_maitresse.pva.create_small_ortho_numpy(c_centre_epip, l_centre_epip, self.projet.size_small_bd_ortho, 1)
        image_maitresse_epip = np.squeeze(image_maitresse_epip)
        # On met à jour l'objet Pva de l'image esclave en définissant sa géométrie épipolaire
        ortho_locale.pva.set_geom_epip(epipGeom, epipGeom.r2e, epipGeom.image2)

        # On cherche le point de corrélation
        correlationEngine = CorrelationEngine(ortho_locale.pva, self.projet.size_small_bd_ortho, image_maitresse_epip)
        # Dans le run de correlationEngine, les coordonnées de la droite sur laquelle chercher le point de corrélation doivent être en géométrie épipolaire
        # On convertit donc c_image et l_image en géométrie épipolaire
        c_epip, l_epip = epipGeom.image_to_epip(c_image, l_image, epipGeom.image2, epipGeom.r2e, use_dh=False)

        # On récupère un c_chap_epip, l_chap_epip en coordonnées épipolaires
        c_chap_epip, l_chap_epip, correlation_max = correlationEngine.run(c_epip, l_epip)

        # On convertit ces coordonnées épipolaires en coordonnées images
        c_chap, l_chap = epipGeom.epip_to_image(c_chap_epip, l_chap_epip, epipGeom.image2, epipGeom.r2e, use_dh=False)


        # Pour contrôle
        # Sauvegarde la vignette maitresse
        save_image(os.path.join(ortho_locale.path_save_pi, "image_maitresse_{}_epipolaire.tif".format(ortho_locale.shot.image)), image_maitresse_epip)

        # Sauvegarde la vignette esclave pour laquelle le score de corrélation est le plus élevé
        image_trouvee_epip = ortho_locale.pva.create_small_ortho_numpy(np.array([[c_chap_epip]]), np.array([[l_chap_epip]]), self.projet.size_small_bd_ortho, 1)
        save_image(os.path.join(ortho_locale.path_save_pi, ortho_locale.shot.image+"_epipolaire_{}.tif".format(str(int(100*correlation_max)))), image_trouvee_epip)

        # On convertit ces coordonnées images en coordonnées terrain
        try :
            x_chap, y_chap, z_chap = ortho_locale.shot.imc.image_to_world(c_chap, l_chap, self.projet.ta.project.dem)
        except:
            x_chap, y_chap, z_chap = 0, 0, 0
            correlation_max = 0
        return x_chap, y_chap, z_chap, correlation_max


    def correlation_orthoLocale(self, ortho_locale, x_world, y_world, reference_ortho):
        
        correlationEngine = CorrelationEngine(ortho_locale, self.projet.size_small_bd_ortho, reference_ortho, v2=False)
        x_chap, y_chap, correlation_max = correlationEngine.run(x_world, y_world)
        z_chap = self.projet.ta.project.dem.get(x_chap, y_chap).item()
        return x_chap, y_chap, z_chap, correlation_max


    def correlation_pva(self, ortho_locale, c_image, l_image, reference_ortho):
        correlationEngine = CorrelationEngine(ortho_locale.pva, self.projet.size_small_bd_ortho, reference_ortho)
        j_chap, i_chap, correlation_max = correlationEngine.run(c_image, l_image)

        # On sauvegarde la vignette qui corrèle le mieux
        ortho_locale.pva.write_vignette(np.array([j_chap]), np.array([i_chap]), 201, os.path.join(ortho_locale.path_save_pi, ortho_locale.shot.image+"_pva.tif"))
        ortho_locale.i_j = (i_chap, j_chap)
        # On convertit les coordonnées images en coordonnées terrain
        try:
            x_chap, y_chap, z_chap = ortho_locale.shot.imc.image_to_world(j_chap, i_chap, self.projet.ta.project.dem)
        except:
            return 0,0,0,0
        return x_chap, y_chap, z_chap, correlation_max



    def compute_correl_epip_pool(self, ortho_locale:OrthoLocale, z_min:float, z_max:float, reference_ortho=None) -> None:
        """
        Sur toutes les orthos locales qui ne sont pas maîtresses, on cherche le point de corrélation sur la droite épipolaire
        """
        
        #On parcourt toutes les orthos locales qui ne sont pas l'image maîtresse
        if ortho_locale != self.image_maitresse:
            #On construit un sous-échantillonnage de la droite entre le sommet de prise de vue et le point terrain de la corrélation
            points = ortho_locale.get_droite_sous_ech(self.image_maitresse, z_min, z_max)
            #On reprojette la droite sur l'image épipolaire
            c_image, l_image = ortho_locale.shot.imc.world_to_image(points[:,0], points[:,1], points[:,2])
            #On reprojette en coordonnées du monde la droite (utile seulement si c'est de la corrélation sur ortho)
            try:
                x_world, y_world, _ = ortho_locale.shot.imc.image_to_world(c_image, l_image, self.projet.ta.project.dem)
                ortho_locale.epip = [Point(x_world[0], y_world[0]), Point(x_world[-1], y_world[-1])]
            except Exception as e:
                print_log("Exception : {}".format(e))
                print_log("Exception : c_image, l_image : {}, {}, {}".format(c_image, l_image, self.projet.ta.project.dem))

            #On sauvegarde les droites épipolaires sur pva et sur ortho locale
            if self.sauvegarde:
                self.save_epip(ortho_locale, c_image, l_image, x_world, y_world)

            if self.type_correlation == "pvaGeomEpip":
                # La corrélation se fait sur les pvas en géométrie épipolaire
                x_chap, y_chap, z_chap, correlation_max = self.correlation_pvaGeomEpip(ortho_locale, c_image, l_image)
  
            elif self.type_correlation == "orthoLocale":
                # On fait la corrélation sur les ortho locales
                x_chap, y_chap, z_chap, correlation_max = self.correlation_orthoLocale(ortho_locale, x_world, y_world, reference_ortho)

            elif self.type_correlation == "pva":
                # On fait la corrélation sur les pvas sans passer par la géométrie épipolaire
                x_chap, y_chap, z_chap, correlation_max = self.correlation_pva(ortho_locale, c_image, l_image, reference_ortho)
                            
            ortho_locale.correlation = correlation_max
            
            ortho_locale.ground_terrain = Point(x_chap, y_chap, z_chap)

            if self.sauvegarde:
                ortho_locale.save_correlation("epip")
        

    def est_en_dehors(self, size, point_x, point_y):
        if point_x < 0 or point_x > size:
            return True
        if point_y < 0 or point_y > size:
            return True
        return False


    def compute_correl_epip(self, z_min:float, z_max:float) -> None:
        """
        Sur toutes les orthos locales qui ne sont pas maîtresses, on cherche le point de corrélation sur la droite épipolaire
        """

        if self.type_correlation == "pvaGeomEpip":
            reference_ortho = None
        elif self.type_correlation == "orthoLocale":
            reference_ortho = self.small_bd_ortho.bd_ortho[0,:,:]
        elif self.type_correlation == "pva":
            point_terrain_correlation = self.image_maitresse.ground_terrain
            x_image, y_image = self.image_maitresse.shot.imc.world_to_image(point_terrain_correlation.x, point_terrain_correlation.y, point_terrain_correlation.z)
            reference_ortho = self.image_maitresse.pva.create_small_ortho_numpy(np.array([x_image]), np.array([y_image]), self.projet.size_small_bd_ortho, 1)
            if self.sauvegarde:
                self.image_maitresse.pva.write_vignette(np.array([x_image]), np.array([y_image]), 201,  os.path.join(self.image_maitresse.path_save_pi, self.image_maitresse.shot.image+"_maitresse_pva.tif"))
        else:
            print("Erreur, self.type_correlation doit avoir une des valeurs suivantes : pvaGeomEpip, orthoLocale ou pva")

        
        for ortho_locale in self.ortho_locales:
            self.compute_correl_epip_pool(ortho_locale, z_min, z_max, reference_ortho=reference_ortho)
        
        self.ortho_locales = sorted(self.ortho_locales, key=lambda d: d.correlation)
        self.ortho_locales.reverse()


    def compute_least_squares(self) -> Tuple[float, float, float, float]:
        """
        Calcule la pseudo-intersection à l'aide des moindres carrés
        """
        system = self.ortho_locales[0].shot.imc.system

        # On calcule la pseudo-intersection par moindres carrés
        #n = len(ortho_locales_valides)
        n = self.get_nb_orthos_valides()
        A = np.zeros((2*n, 3))
        B = np.zeros((2*n, 1))

        compte = 0
        for orthoLocale in self.ortho_locales:
            if orthoLocale.calcul_valide:

                # On construit une base orthonormée
                gt = orthoLocale.get_ground_terrain()
                gt_geoc = np.array(system.world_to_euclidean(gt[0], gt[1], gt[2]))

                gs = orthoLocale.get_sommet()
                gs_geoc = np.array(system.world_to_euclidean(gs[0], gs[1], gs[2]))
                u = gt_geoc - gs_geoc
                u /= np.linalg.norm(u)
                v = np.random.uniform(size=(3))
                v -= u.dot(v) * u
                v /= np.linalg.norm(v)
                w = np.cross(u, v)
                w /= np.linalg.norm(w)

                A[2*compte, 0] = v[0]
                A[2*compte, 1] = v[1]
                A[2*compte, 2] = v[2]

                A[2*compte+1, 0] = w[0]
                A[2*compte+1, 1] = w[1]
                A[2*compte+1, 2] = w[2]

                B[2*compte, 0] = np.sum(gs_geoc * v)
                B[2*compte+1, 0] = np.sum(gs_geoc * w)
                compte += 1

        x_chap, res, _, _ = np.linalg.lstsq(A, B, rcond=None)
        resultat = system.euclidean_to_world(x_chap[0], x_chap[1], x_chap[2])

        print_log("L'intersection est en ({}, {}, {}). Nombre d'images : {}".format(resultat[0].item(), resultat[1].item(), resultat[2].item(), n))
        print_log("Le résidu est de {} mètres".format(res.item()))
        return res.item(), resultat[0].item(), resultat[1].item(), resultat[2].item()


    def compute_decalage(self) -> None:
        """
        Méthode par décalage : la corrélation ne se fait plus sur le point initial, mais un peu décalé sur 
        la ligne de visée pour ne plus être gêné par le sursol
        """

        # On récupère le centre de corrélation de l'image maitresse
        image_maitresse = self.ortho_locales[0]
        centre = image_maitresse.ground_terrain

        # On détermine le carré sur lequel se trouvera le nouveau point à corréler
        x_int_0 = centre.x - self.projet.size_bd_ortho/4 * self.resolution
        y_int_0 = centre.y + self.projet.size_bd_ortho/4 * self.resolution
        x_int_1 = centre.x + self.projet.size_bd_ortho/4 * self.resolution
        y_int_1 = centre.y - self.projet.size_bd_ortho/4 * self.resolution

        #s = geopandas.GeoSeries([LineString([
        #    (x_int_0, y_int_0), 
        #    (x_int_1, y_int_0),
        #    (x_int_1, y_int_1), 
        #    (x_int_0, y_int_1), 
        #    (x_int_0, y_int_0)])
        #],)

        #On projette la ligne de visée au sol
        ligne_visee = LineString([(centre.x, centre.y), (image_maitresse.get_nadir()[0], image_maitresse.get_nadir()[1])])

        #On intersecte la ligne de visée avec le carré
        intersection = s.intersection(ligne_visee)
        if len(intersection) != 1:
            print_log("Curieux : il n'y a pas une unique intersection mais {}".format(len(intersection)))
        centre_vignette = intersection[0]

        #On construit la nouvelle ortho centrée sur ce nouveau point
        self.bd_ortho_decalage = Ortho(self.resolution, int((self.projet.size_bd_ortho+1)/2), os.path.join(self.path, "decalage"), centre=centre_vignette, path_ortho=self.projet.ortho)
        self.bd_ortho_decalage.create_ortho()
        self.bd_ortho_decalage.save_ortho()
        
        #On calcule le point de corrélation
        self.compute_correlations(self.bd_ortho_decalage, "dec")

        #On supprime les pvas pour lesquelles la valeur de corrélation est trop faible
        self.filter_ortho_locales(self.projet.seuil_ortho_locale, self.ortho_locales_valides_dec)
        
        #On calcule la pseudo-intersection
        solution, nb_images, residus = self.compute_pseudo_intersection(self.ortho_locales_valides_dec)
        z_chap = solution[2]

        # On récupère le point sol
        lamb = (z_chap - image_maitresse.get_sommet()[2]) / (image_maitresse.get_ground_terrain()[2] - image_maitresse.get_sommet()[2])
        x_chap = image_maitresse.get_sommet()[0] + lamb * (image_maitresse.get_ground_terrain()[0] - image_maitresse.get_sommet()[0])
        y_chap = image_maitresse.get_sommet()[1] + lamb * (image_maitresse.get_ground_terrain()[1] - image_maitresse.get_sommet()[1])
        self.x_chap_dec = np.array([x_chap, y_chap, z_chap])



    def create_nav_csv(self) -> None:
        """
        Crée un fichier navigation.csv qui contient les sommets de prise de vue et leurs orientations
        """

        with open(os.path.join(self.path, "micmac", "navigation.csv"), "w") as f:

            # Ecriture de l'en-tête
            f.write("#F=N X Y Z K W P\n#\n##image latitude longitude altitude Kappa Omega Phi\n")
            # On parcourt toutes les pvas concernant le point
            for ortho_locale in self.ortho_locales:
                shot = ortho_locale.shot
                nom = shot.image  # A faire en plus propre ?
                
                opk = shot.imc.system.mat_to_opk(shot.imc.mat)
                omega = opk[0]
                phi = opk[1]
                kappa = opk[2]

                image_conical = shot.imc
                scale_factor = image_conical.system.proj_engine.get_scale_factor(image_conical.x_pos, image_conical.y_pos)
                z_pos_cor = image_conical.z_pos + scale_factor * (image_conical.z_pos - self.projet.ta.project.dem.get(image_conical.x_pos, image_conical.y_pos))

                # Pour chaque pva, on écrit dans le fichier csv les coordonnées du sommet de prise de vue et les orientations
                f.write("{} {} {} {} {} {} {}\n".format(nom+".tif", image_conical.x_pos, image_conical.y_pos, z_pos_cor, kappa, omega, phi))


    def get_sensor_size(self) -> Tuple[int, int]:
        """
        Il faut que toutes les images aient la même taille. On récupère la taille de l'extrait d'image la plus grande
        """
        
        height_max = 0
        width_max = 0
        for ortho_locale in self.ortho_locales:

            height_max = max(height_max, ortho_locale.max_l - ortho_locale.min_l)
            width_max = max(width_max, ortho_locale.max_c - ortho_locale.min_c)

        return height_max, width_max

    def crop_images(self, height_max: int, width_max: int) -> None:
        """
        Découpe les pvas pour ne garder que les extraits nécessaires autour du point à calculer
        """
        
        for ortho_locale in self.ortho_locales:
            inputds = gdal.Open(os.path.join("chantiers", "49_2020", "pva", ortho_locale.shot.image+".jp2"))
            image = inputds.ReadAsArray(ortho_locale.min_c, ortho_locale.min_l, width_max+1, height_max+1)
            inputds = None
            save_image(os.path.join(self.path, "micmac", ortho_locale.shot.image+".tif"), image)

    def create_MicMac_LocalChantierDescripteur(self) -> Tuple[float, float, float]:
        """
        Crée un fichier MicMac-LocalChantierDescripteur.xml dont MicMac a besoin pour calculer un fichier d'orientation
        """

        # On récupère un template
        tree = etree.parse(os.path.join("scripts", "MicMac-LocalChantierDescripteur.xml"))
        root = tree.getroot()

        # On récupère la caméra du chantier
        camera_keys = list(self.projet.ta.project.camera)
        camera_name = camera_keys[0]
        camera = self.projet.ta.project.camera[camera_name]

        # On modifie le nom de la caméra
        root.find(".//Name").text = "{}".format(camera_name)

        # On modifie la taille des capteurs
        SzCaptMm_h = camera._h * camera._pixel_size * 1000
        SzCaptMm_w = camera._w * camera._pixel_size * 1000
        root.find(".//SzCaptMm").text = "{} {}".format(SzCaptMm_w, SzCaptMm_h)

        # On modifie la focale
        CalcName = camera._focal * camera._pixel_size * 1000
        calcNameFindAll = root.findall(".//CalcName")
        calcNameFindAll[0].text = "{}".format(camera_name)
        calcNameFindAll[1].text = "{}".format(CalcName)

        # On sauvegarde le fichier xml
        with open(os.path.join(self.path, "micmac", "MicMac-LocalChantierDescripteur.xml"), "w") as f:
            f.write(str(etree.tostring(tree, encoding='unicode')))

        return camera._focal, camera._x_ppa, camera._y_ppa

    
    def update_ori(self, height_max: int, width_max: int, focale: float, x_ppa: int, y_ppa: int) -> None:
        """
        Modifie le fichier orientation pour qu'il soit adapté aux images réduites
        """
        
        for ortho_locale in self.ortho_locales:

            # On ouvre le fichier orientation d'une pva
            tree = etree.parse(os.path.join(self.path, "micmac", "Ori-Nav-Crop", "Orientation-{}.tif.xml".format(ortho_locale.shot.image)))
            root = tree.getroot()

            # On corrige la position du PP
            pp_balise = root.find(".//PP")
            pp_balise.text = "{} {}".format(x_ppa - ortho_locale.min_c, y_ppa - ortho_locale.min_l)

            # On corrige la taille de l'image
            root.find(".//SzIm").text = "{} {}".format(width_max+1, height_max+1)

            # On corrige la position du centre de distorsion
            root.find(".//CDist").text = "{} {}".format(x_ppa - ortho_locale.min_c, y_ppa - ortho_locale.min_l)

            # On corrige la focale
            root.find(".//F").text = "{}".format(focale)

            #Utile ??
            ModRad = root.find(".//ModRad")
            CalibDistortion = root.find(".//CalibDistortion")
            CalibDistortion.remove(ModRad)
            ModNoDist = etree.SubElement(CalibDistortion, "ModNoDist")
            Inutile = etree.SubElement(ModNoDist, "Inutile")
            Inutile.text = " "
           
            # On sauvegarde le fichier orientation
            with open(os.path.join(self.path, "micmac", "Ori-Nav-Crop", "Orientation-{}.tif.xml".format(ortho_locale.shot.image)), "w") as f:
                f.write(str(etree.tostring(tree, encoding='unicode')))

    def interpolate(self, image_maitresse: OrthoLocale) -> float:
        """
        Cherche les coordonnées 3D du point terrain à partir du fichier généré par MicMac
        """
        # On ouvre le fichier NuageImProf_STD-MALT_Etape_6_XYZ.tif
        # Ce fichier contient trois bandes : x, y et z. Avec l'aide de ces points, on peut interpoler
        inputds = gdal.Open(os.path.join(self.path, "micmac", "MEC-Malt", "NuageImProf_STD-MALT_Etape_6_XYZ.tif"))

        image = inputds.ReadAsArray()

        x = image[0, :, :].reshape((-1, 1))
        y = image[1, :, :].reshape((-1, 1))
        z = image[2, :, :].reshape((-1, 1))

        points = np.concatenate((x, y), axis=1)

        z_alti = griddata(points, z, (image_maitresse.get_ground_terrain()[0], image_maitresse.get_ground_terrain()[1]), method='linear')
        return z_alti


    def compute_MicMac(self) -> None:
        """
        Recherche les coordonnées 3D du point avec la reconstruction 3D de MicMac
        """

        if not os.path.exists(os.path.join(self.path, "micmac")):
            os.makedirs(os.path.join(self.path, "micmac"))

        # On crée le fichier csv qui permettra à MicMac de créer un fichier orientation
        self.create_nav_csv()

        # On récupère la taille des imagettes
        height_max, width_max = self.get_sensor_size()

        # On découpe les images
        self.crop_images(height_max, width_max)

        # On crée le fichier MicMac_LocalChantierDescripteur
        focale, x_ppa, y_ppa = self.create_MicMac_LocalChantierDescripteur()

        # L'altitude du nadir étant souvent à 0, il faut la recalculer. MicMac en a besoin dans le fichier orientation
        image_maitresse = self.ortho_locales[0]
        z_nadir = self.projet.ta.project.dem.get(image_maitresse.get_nadir()[0], image_maitresse.get_nadir()[1]).item()

        # On crée le fichier orientation MicMac
        commande_export = 'mm3d OriConvert OriTxtInFile navigation.csv Nav AltiSol={} >> logfile'.format(z_nadir)
        os.system("cd {}\n{}".format(os.path.join(self.path, "micmac"), commande_export))

        shutil.copytree(os.path.join(self.path, "micmac", "Ori-Nav"),
                        os.path.join(self.path, "micmac", "Ori-Nav-Crop"))

        # On modifie les fichiers orientations car on a gardé seulement un extrait des images
        self.update_ori(height_max, width_max, focale, x_ppa, y_ppa)


        # On calcule la carte de profondeur avec Malt
        nom_images = "|".join([i.shot.image + ".tif" for i in self.ortho_locales])
        commande_Malt = 'mm3d Malt GeomImage "{}" Nav-Crop Master="{}" DirMEC=MEC-Malt >> logfile'.format(nom_images, image_maitresse.shot.image+".tif")

        os.system("cd {}\n{}".format(os.path.join(self.path, "micmac"), commande_Malt))

        # On crée un nuage de point et un fichier raster à la partir de la carte de profondeur de Malt
        commande_Nuage2Ply = "mm3d Nuage2Ply MEC-Malt/NuageImProf_STD-MALT_Etape_6.xml Attr={} DoXYZ=1 >> logfile".format(image_maitresse.shot.image+".tif")

        os.system("cd {}\n{}".format(os.path.join(self.path, "micmac"), commande_Nuage2Ply))

        # On interpole l'altitude à partir des coordonnées du point sur l'image de référence
        z_chap = self.interpolate(image_maitresse).item()

        self.x_chap_micmac = np.array([image_maitresse.get_ground_terrain()[0], image_maitresse.get_ground_terrain()[1], z_chap])



    def print_results(self, micmac: bool, decalage: bool) -> None:
        """
        Affiche les écarts entre les méthodes et avec les coordonnées plani initiales du point
        """
        print_log("Méthode par intersection : {}".format(self.x_chap))
        if decalage:
            print_log("Méthode par décalage : {}".format(self.x_chap_dec))
        if micmac:
            print_log("Méthode par MicMac : {}".format(self.x_chap_micmac))
        print_log("")

        print_log("intersection/point_init, dh : {} mètres".format(np.sqrt((self.x_chap[0] - self.point.x)**2 + (self.x_chap[1] - self.point.y)**2)))
        if decalage:
            print_log("décalage/point_init, dh : {} mètres".format(np.sqrt((self.x_chap_dec[0] - self.point.x)**2 + (self.x_chap_dec[1] - self.point.y)**2)))
        if micmac:
            print_log("micmac/point_init, dh : {} mètres".format(np.sqrt((self.x_chap_micmac[0] - self.point.x)**2 + (self.x_chap_micmac[1] - self.point.y)**2)))
        print_log("")
        
        
        if decalage:
            print_log("intersection/décalage, dh : {} mètres".format(np.sqrt((self.x_chap[0] - self.x_chap_dec[0])**2 + (self.x_chap[1] - self.x_chap_dec[1])**2)))
            print_log("intersection/décalage, dz : {} mètres".format(self.x_chap_dec[2] - self.x_chap[2]))
            print_log("")

        if micmac:
            print_log("intersection/micmac, dh : {} mètres".format(np.sqrt((self.x_chap[0] - self.x_chap_micmac[0])**2 + (self.x_chap[1] - self.x_chap_micmac[1])**2)))
            print_log("intersection/micmac, dz : {} mètres".format(self.x_chap_micmac[2] - self.x_chap[2]))
            print_log("")

        if decalage and micmac:
            print_log("décalage/micmac, dh : {} mètres".format(np.sqrt((self.x_chap_dec[0] - self.x_chap_micmac[0])**2 + (self.x_chap_dec[1] - self.x_chap_micmac[1])**2)))
            print_log("décalage/micmac, dz : {} mètres".format(self.x_chap_micmac[2] - self.x_chap_dec[2]))
            print_log("")


    def add_point_ply(self, point: np.array, couleur: Tuple[int, int, int]) -> None:
        """
        Ajoute le point dans le fichier ply généré par MicMac
        """

        if point[0] != 0 and point[1] != 0:

            with open(os.path.join(self.path, "resultat.ply"), 'rb') as f:
                plydata = PlyData.read(f)
            
            axe_x = np.linspace(point[0]-1, point[0]+1, num=100).reshape((100, 1))
            axe_x_const = np.ones((100, 1)) * point[0]

            axe_y = np.linspace(point[1]-1, point[1]+1, num=100).reshape((100, 1))
            axe_y_const = np.ones((100, 1)) * point[1]

            axe_z = np.linspace(point[2]-1, point[2]+1, num=100).reshape((100, 1))
            axe_z_const = np.ones((100, 1)) * point[2]
    
            bloc1 = np.concatenate((axe_x, axe_y_const, axe_z_const), axis=1)
            bloc2 = np.concatenate((axe_x_const, axe_y, axe_z_const), axis=1)
            bloc3 = np.concatenate((axe_x_const, axe_y_const, axe_z), axis=1)

            bloc_axes = np.concatenate((bloc1, bloc2, bloc3), axis=0)


            for i in range(bloc_axes.shape[0]):
                new_vertex = np.array((bloc_axes[i,0], bloc_axes[i,1], bloc_axes[i,2], couleur[0], couleur[1], couleur[2]), dtype=plydata['vertex'].data.dtype) 
                plydata['vertex'].data = np.r_[plydata['vertex'].data, new_vertex]

            
            with open(os.path.join(self.path, "resultat.ply"), mode='wb') as f:
                plydata.write(f)

    def get_liste_meme_bande(self)->List[str]:
        liste = []
        image_maitresse = self.image_maitresse.shot.image
        for flight in self.projet.ta.project.get_flights():
            for strip in flight.get_strips():
                present = False
                for shot in strip.get_shots():
                    if shot.image == image_maitresse:
                        present = True
                if present:
                    for shot in strip.get_shots():
                        liste.append(shot.image)
        return liste