import os
import shutil
from shapely import contains
import numpy as np
from .ortho import Ortho
from .orthoLocale import OrthoLocale
from typing import List, Tuple
from shapely.geometry import LineString, Point
import geopandas
from .tool import print_log, save_image
from .correlation_engine import CorrelationEngine
from .epipolarGeometry import EpipolarGeometry
from .shot import Shot

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
            if contains(shot.emprise, self.point):
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
            orthoLocale = OrthoLocale(self.resolution, self.point, pva, self.projet.size_orthoLocale, os.path.join(self.projet.pva, pva.image+".tif"), self.projet, os.path.join(self.path, "orthos_locales"), os.path.join(self.path, "decalage"))
            ortho = None
            if contains(pva.emprise, orthoLocale.emprise):
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


    def correlation_pvaGeomEpip(self, ortho_locale:Shot, c_image, l_image):
        # On crée l'objet EpipolarGeometry qui calcule immédiatement les matrices pour passer de géométrie image à géométrie épipolaire
        epipGeom = EpipolarGeometry(self.image_maitresse.shot, ortho_locale.shot, self.projet.mnt, self.projet.pva)

        # On construit la vignette maitresse à partir de la géométrie épipolaire de l'image maitresse 
        z = self.projet.mnt.get(int(self.small_bd_ortho.centre.x), int(self.small_bd_ortho.centre.y))
        centre_pva = self.image_maitresse.shot.world_to_image(self.small_bd_ortho.centre.x, self.small_bd_ortho.centre.y, z)
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
            x_chap, y_chap, z_chap = ortho_locale.shot.image_to_world(c_chap, l_chap, self.projet.mnt)
        except:
            x_chap, y_chap, z_chap = 0, 0, 0
            correlation_max = 0
        return x_chap, y_chap, z_chap, correlation_max


    def correlation_orthoLocale(self, ortho_locale, x_world, y_world, reference_ortho):
        
        correlationEngine = CorrelationEngine(ortho_locale, self.projet.size_small_bd_ortho, reference_ortho, v2=False)
        x_chap, y_chap, correlation_max = correlationEngine.run(x_world, y_world)
        z_chap = self.projet.mnt.get(x_chap, y_chap).item()
        return x_chap, y_chap, z_chap, correlation_max


    def correlation_pva(self, ortho_locale, c_image, l_image, reference_ortho):
        correlationEngine = CorrelationEngine(ortho_locale.pva, self.projet.size_small_bd_ortho, reference_ortho)
        j_chap, i_chap, correlation_max = correlationEngine.run(c_image, l_image)

        # On sauvegarde la vignette qui corrèle le mieux
        ortho_locale.pva.write_vignette(np.array([j_chap]), np.array([i_chap]), 201, os.path.join(ortho_locale.path_save_pi, ortho_locale.shot.image+"_pva.tif"))
        ortho_locale.i_j = (i_chap, j_chap)
        # On convertit les coordonnées images en coordonnées terrain
        try:
            x_chap, y_chap, z_chap = ortho_locale.shot.image_to_world(j_chap, i_chap, self.projet.mnt)
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
            c_image, l_image = ortho_locale.shot.world_to_image(points[:,0], points[:,1], points[:,2])
            #On reprojette en coordonnées du monde la droite (utile seulement si c'est de la corrélation sur ortho)
            try:
                x_world, y_world, _ = ortho_locale.shot.image_to_world(c_image, l_image, self.projet.mnt)
                ortho_locale.epip = [Point(x_world[0], y_world[0]), Point(x_world[-1], y_world[-1])]
            except Exception as e:
                print_log("Exception : {}".format(e))
                print_log("Exception : c_image, l_image : {}, {}, {}".format(c_image, l_image, self.projet.mnt))

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
            x_image, y_image = self.image_maitresse.shot.world_to_image(point_terrain_correlation.x, point_terrain_correlation.y, point_terrain_correlation.z)
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
                gt_geoc = np.array(orthoLocale.shot.world_to_euclidean(gt[0], gt[1], gt[2]))

                gs = orthoLocale.get_sommet()
                gs_geoc = np.array(orthoLocale.shot.world_to_euclidean(gs[0], gs[1], gs[2]))
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
        resultat = orthoLocale.shot.euclidean_to_world(x_chap[0], x_chap[1], x_chap[2])

        print_log("L'intersection est en ({}, {}, {}). Nombre d'images : {}".format(resultat[0].item(), resultat[1].item(), resultat[2].item(), n))
        print_log("Le résidu est de {} mètres".format(res.item()))
        return res.item(), resultat[0].item(), resultat[1].item(), resultat[2].item()


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