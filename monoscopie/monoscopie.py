from shapely.geometry import Point
import os
from .chantier import Chantier
from osgeo import gdal
from typing import List
import time
from .tool import print_log 
from .infosResultats import InfosResultats
from .orthoLocale import OrthoLocale
from .shot import Shot, MNT, RAF
from lxml import etree


class Monoscopie:
    """
    A class for getting x, y, z coordinates from points on BD Ortho
    """

    def __init__(self, pva: str, ortho: str, mnt: str, ta_xml: str, resultats: str, size_orthoLocale=141, size_bd_ortho=61, size_small_bd_ortho=11, seuil_maitresse=0.9, seuil_ortho_locale=0.4, log=False, micmac=False, decalage=False, type_correlation="pva", sauvegarde=False) -> None:
        self.pva = pva
        self.ortho = ortho
        self.mnt = MNT(mnt)
        self.ta_xml = ta_xml
        self.resultats = resultats
        self.points: List[Point] = []
        self.size_orthoLocale = size_orthoLocale
        self.size_bd_ortho = size_bd_ortho
        self.seuil_maitresse = seuil_maitresse
        self.seuil_ortho_locale = seuil_ortho_locale
        plugin_path = os.path.dirname(os.path.realpath(__file__))
        self.raf = RAF(os.path.join(plugin_path, "raf2020_2154.tif"))
        self.size_small_bd_ortho = size_small_bd_ortho
        self.log = log
        self.micmac = micmac
        self.decalage = decalage
        self.type_correlation = type_correlation
        self.id = 0
        self.chantier = None
        self.infosResultats = None
        self.sauvegarde = sauvegarde

        self.shots = []
        
        self.get_shots()

        # On récupère la résolution de l'ortho
        self.get_resolution()


    
    def getFocale(self, root):
        focal = root.find(".//focal")
        focale_x = float(focal.find(".//x").text)
        focale_y = float(focal.find(".//y").text)
        focale_z = float(focal.find(".//z").text)
        return [focale_x, focale_y, focale_z]

    def get_centre_rep_local(self, root):
        centre_rep_local = root.find(".//centre_rep_local")
        centre_rep_local_x = float(centre_rep_local.find(".//x").text)
        centre_rep_local_y = float(centre_rep_local.find(".//y").text)
        return [centre_rep_local_x, centre_rep_local_y]


    def get_shots(self):
        tree = etree.parse(self.ta_xml)
        root = tree.getroot()
        focale = self.getFocale(root)
        centre_rep_local = self.get_centre_rep_local(root)
        pvas = [i.split(".")[0] for i in os.listdir(self.pva)]
        for cliche in root.getiterator("cliche"):
            image = cliche.find("image").text.strip()
            if image in pvas:
                shot = Shot.createShot(cliche, focale, os.path.join(self.pva, "{}.tif".format(image)), self.raf, centre_rep_local)
                self.shots.append(shot)




    
    def get_resolution(self) -> None:
        """ 
        On récupère la résolution de la bd ortho
        """
        inputds = gdal.Open(self.ortho)
        geotransform = inputds.GetGeoTransform()
        self.resolution = geotransform[1]
        print_log("Résolution de l'ortho : {}".format(self.resolution))
    


    def run(self, point:Point, orthoLocaleMaitresse:OrthoLocale=None, z_min:float=None, z_max:float=None, meme_bande=False):
        """
        Détermine les coordonnées x, y, z des points
        """

        print_log("Point : {}".format(point))

        tic = time.time()
        #On construit pour chaque point la classe Chantier
        self.chantier = Chantier(point, self.id, self.resolution, self, type_correlation=self.type_correlation, sauvegarde=self.sauvegarde)
        
        #On récupère les pvas dont l'emprise contient le point
        self.chantier.get_pvas(self.shots)
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
        success = self.chantier.improve_correlations()
        if not success:
            self.infosResultats = InfosResultats(False)
        else:
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
        z = self.mnt.get(self.chantier.point.x, self.chantier.point.y)
        print_log("résultat final : {}".format(x_chap))
        if nb_images == 0:
            self.infosResultats = InfosResultats(False)
        else:
            self.infosResultats = InfosResultats(True, self.id, self.chantier.point, z, x_chap, nb_images, residus)
        
        