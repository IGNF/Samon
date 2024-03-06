from monoscopie.monoscopie import Monoscopie
from shapely.geometry import Point
import numpy as np
import time
from tqdm import tqdm



# répertoire contenant les pvas
pva = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/pvas"

# fichier vrt regroupant les dalles de la BD Ortho
ortho = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/ortho/ortho.vrt"

# fichier vrt regroupant les dalles du MNT
mnt =  "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/mnt/mnt.vrt"

# tableau d'assemblage sous format xml
ta_xml = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/Orientation/22FD0520_adjust.XML"

# répertoire où enregistrer les données intermédiaires et les résultats des calculs
resultats = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_plugin"

# grille raf
raf = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/raf/raf2020_2154.tif"


point = Point(987159.5, 6403364)


z_min = 1025
z_max = 1070

monoscopie = Monoscopie(pva, ortho, mnt, ta_xml, resultats, raf, size_small_bd_ortho=11, type_correlation="pvaGeomEpip")
monoscopie.run(point, z_min=z_min, z_max=z_max)