from pysocle.photogrammetry.ta import Ta
from shapely.geometry import Point
import os
from .chantier import Chantier
from osgeo import gdal
from typing import List
import time
from .tool import print_log 
from .infosResultats import InfosResultats
from .orthoLocale import OrthoLocale
from tqdm import tqdm

class Monoscopie:
    """
    A class for getting x, y, z coordinates from points on BD Ortho
    """

    def __init__(self, pva: str, ortho: str, mnt: str, ta_xml: str, resultats: str, raf: str, size_orthoLocale=141, size_bd_ortho=61, size_small_bd_ortho=5, seuil_maitresse=0.9, seuil_ortho_locale=0.4, log=False, micmac=False, decalage=False, type_correlation="pva", sauvegarde=False) -> None:
        self.pva = pva
        self.ortho = ortho
        self.mnt = mnt
        self.ta_xml = ta_xml
        self.resultats = resultats
        self.points: List[Point] = []
        self.size_orthoLocale = size_orthoLocale
        self.size_bd_ortho = size_bd_ortho
        self.seuil_maitresse = seuil_maitresse
        self.seuil_ortho_locale = seuil_ortho_locale
        self.raf = raf
        self.size_small_bd_ortho = size_small_bd_ortho
        self.log = log
        self.micmac = micmac
        self.decalage = decalage
        self.type_correlation = type_correlation
        self.id = 0
        self.chantier = None
        self.infosResultats = None
        self.sauvegarde = sauvegarde

        # On charge le fichier ta_xml
        self.ta = Ta.from_xml(self.ta_xml)
        print_log("Fichier xml chargé")

        # On ajoute le MNT au projet
        self.ta.project.add_dem(self.mnt)
        print_log("MNT ajouté")

        # On récupère la résolution de l'ortho
        self.get_resolution()

        # On retire les images qui ne sont pas dans le répertoire pva
        self.remove_shot()

        # Dans les fichiers TA, ce sont des hauteurs ellipsoïdales. Il faut les convertir en altitude
        self.ta.project.conversion_elevation(self.raf, "a")

        
        print_log("Calcul des emprises au sol des clichés")
        for shot in tqdm(self.ta.project.get_shots()):
            shot.compute_extent(self.ta.project.dem, 4, 1, 2)
        print_log("Emprises au sol calculées")


        # Facultatif : on exporte les emprises des clichés
        #self.ta.save_extent_shot("test.shp")

    
    def get_resolution(self) -> None:
        """ 
        On récupère la résolution de la bd ortho
        """
        inputds = gdal.Open(self.ortho)
        geotransform = inputds.GetGeoTransform()
        self.resolution = geotransform[1]
        print_log("Résolution de l'ortho : {}".format(self.resolution))
    

    def remove_shot(self) -> None:
        """
        On retire de la liste des shots tous les shots dont les pvas correspondantes sont manquantes
        """
        pvas = [i.split(".")[0] for i in os.listdir(self.pva)]
        compte = 0
        for flight in self.ta.project.get_flights():
            for strip in flight.get_strips():
                shot_to_remove = []
                # On récupère la liste des images qui ne sont pas dans le répertoire pva
                for shot in strip.get_shots():
                    if shot.image not in pvas:
                        compte += 1
                        shot_to_remove.append(shot)
                # On supprime ces images
                for shot in shot_to_remove:
                    strip.remove_shot(shot)
        print_log("{} images ont été retirées. Il en reste {}.".format(compte, self.ta.project.nbr_shot()))   


    def run(self, point:Point, orthoLocaleMaitresse:OrthoLocale=None, z_min:float=None, z_max:float=None, meme_bande=False):
        """
        Détermine les coordonnées x, y, z des points
        """

        print_log("Point : {}".format(point))

        tic = time.time()
        #On construit pour chaque point la classe Chantier
        self.chantier = Chantier(point, self.id, self.resolution, self, type_correlation=self.type_correlation, sauvegarde=self.sauvegarde)
        
        #On récupère les pvas dont l'emprise contient le point
        self.chantier.get_pvas(self.ta.project.get_shots())
        print_log("get_pvas : {}".format(time.time()-tic))
        #S'il n'y a pas au moins deux pvas, alors on passe au point suivant
        if len(self.chantier.pvas) < 2:
            self.infosResultats = InfosResultats(False)
        #On crée les orthos extraites de la BD Ortho
        if orthoLocaleMaitresse:
            self.chantier.create_bd_ortho(orthoLocaleMaitresse.shot)
        else:
            self.chantier.create_bd_ortho()
        print_log("get_bd_ortho : {}".format(time.time()-tic))
        self.chantier.create_small_ortho()
        #Pour chaque pva, on crée des orthos locales
        self.chantier.create_orthos_locales()
        print_log("Ortho locales créées : {}".format(time.time()-tic))
        print_log("\nDébut de la méthode par pseudo-intersection")
        #On calcule grossièrement le point de corrélation entre la BD Ortho et les orthos locales
        self.chantier.compute_correlations(self.chantier.bd_ortho, "pi")
        print_log("compute_correlations : {}".format(time.time()-tic))
        #On affine la corrélation pour les pvas dont le score de corrélation est supérieur à self.seuil_maitresse
        self.chantier.improve_correlations()
        print_log("improve_correlations : {}".format(time.time()-tic))
        #Pour toutes les images qui ne sont pas maitresses, on recherche le point de corrélation sur la droite épipolaire
        self.chantier.compute_correl_epip(z_min, z_max)
        print_log("compute_correl_epip : {}".format(time.time()-tic))

        #Si on est dans le mode meme_bande, alors on récupère
        #toutes les images du même axe de vol que l'image maîtresse
        liste_meme_bande = []
        if meme_bande:
            liste_meme_bande = self.chantier.get_liste_meme_bande()
        #On ne conserve que les orthos locales pour lesquelles le score de corrélation est supérieur à self.seuil_ortho_locale
        self.chantier.filter_ortho_locales(self.seuil_ortho_locale, liste_meme_bande)
        #On calcule la pseudo-intersection
        self.lancer_calcul()
        self.id += 1
        print_log("Fin : {}".format(time.time()-tic))
        


    def lancer_calcul(self):
        x_chap, nb_images, residus = self.chantier.compute_pseudo_intersection()
        self.chantier.x_chap = x_chap
        z = self.ta.project.dem.get(self.chantier.point.x, self.chantier.point.y)
        print_log("résultat final : {}".format(x_chap))
        if nb_images == 0:
            self.infosResultats = InfosResultats(False)
        else:
            self.infosResultats = InfosResultats(True, self.id, self.chantier.point, z, x_chap, nb_images, residus)
        
        