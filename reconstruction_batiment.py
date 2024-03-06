from monoscopie.monoscopie import Monoscopie
from shapely.geometry import Point
import numpy as np
import time
from tqdm import tqdm

"""
Calcule tous les points d'une zone et les sauvegarde dans un fichier xyz

La zone est définie par l'angle supérieur gauche, par la taille d'un côté, et la résolution des points dans cette zone
"""

# répertoire contenant les pvas
pva = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/pvas"

# fichier vrt regroupant les dalles de la BD Ortho
ortho = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/ortho_20/ortho.vrt"

# fichier vrt regroupant les dalles du MNT
mnt =  "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/mnt/mnt.vrt"

# tableau d'assemblage sous format xml
ta_xml = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/Orientation/22FD0520_adjust.XML"

# répertoire où enregistrer les données intermédiaires et les résultats des calculs
resultats = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats"

# grille raf
raf = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/raf/raf2020_2154.tif"

# altitude minimale et altitude maximale de recherche des points
z_min = 1025
z_max = 1070

# fichier où sauvegarder les résultats sous format xyz



# type correlation :
#type_correlation = "pvaGeomEpip"
#type_correlation = "orthoLocale"
type_correlation = "pva"

# Définition de la zone
# Angle nord-est de la zone à reconstruire
point_initial = Point(987111,6403475)
#point_initial = Point(987426,6403550) #Sommet de la tour ronde

# Taille en mètres du côté de la zone à reconstruire 
distance = 130
#distance = 6

#Résolution en mètres
resolution = 0.5







def reconstruction(monoscopie:Monoscopie, fichier_xyz:str):

    # Création de la grille de points à calculer
    x = np.arange(point_initial.x, point_initial.x+distance+resolution, resolution)
    y = np.arange(point_initial.y-distance-resolution, point_initial.y, resolution)

    points_bati = []
    tic = time.time()


    compte = 0
    # Parcourt tous les points
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            compte +=1
            # Création d'un objet shapely.Point
            point = Point(x[i], y[j])
            # Calcul des coordonnées du point
            # Si z_min et z_max ne sont pas définis, alors la zone de recherche se fait dans -20 m et + 100 de la valeur du MNT au point cliqué
            monoscopie.run(point, z_min=z_min, z_max=z_max)
            
            # Si le calcul a abouti:
            if monoscopie.infosResultats.reussi:
                # Récupère les informations du calcul dans monoscopie.infosResultats
                points_bati.append([monoscopie.infosResultats.point3d, monoscopie.infosResultats.nb_images])
                #print(monoscopie.infosResultats.point3d)
                #print(monoscopie.infosResultats.nb_images)
                #print(monoscopie.infosResultats.residu)


    # Ouverture du fichier xyz
    with open(fichier_xyz, "w") as f:
        # Parcourt tous les points
        for point_bati in points_bati:
            # Ecrit les coordonnées du point ainsi que le nombre de pvas ayant servi au calcul
            f.write("{} {} {} {}\n".format(point_bati[0][0], point_bati[0][1], point_bati[0][2], point_bati[1]))

    duree = time.time() - tic
    print("Toc : {}".format(duree))
    print("Temps moyen : {}".format(duree/compte))




# Création de l'objet Monoscopie
type_correlation = "pva"
monoscopie = Monoscopie(pva, ortho, mnt, ta_xml, resultats, raf, size_small_bd_ortho=11, type_correlation=type_correlation, sauvegarde=False)
fichier_xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_bati/bati_pva_11_epip_large.xyz"
reconstruction(monoscopie, fichier_xyz)

type_correlation = "pvaGeomEpip"
monoscopie = Monoscopie(pva, ortho, mnt, ta_xml, resultats, raf, size_small_bd_ortho=11, type_correlation=type_correlation)
fichier_xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_bati/bati_pvaGeomEpip_11.xyz"
reconstruction(monoscopie, fichier_xyz)

type_correlation = "orthoLocale"
monoscopie = Monoscopie(pva, ortho, mnt, ta_xml, resultats, raf, size_small_bd_ortho=11, type_correlation=type_correlation)
fichier_xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_bati/bati_orthoLocale_11.xyz"
reconstruction(monoscopie, fichier_xyz)