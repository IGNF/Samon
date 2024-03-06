import numpy as np
from osgeo import gdal
from scipy import ndimage

mns_lidar_path = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/lidar/MNS.tif"
xyz_propre = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/entree_Mont_Dauphin_nb_images_nettoye.xyz"
xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/entree_Mont_Dauphin_nb_images.xyz"
xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/entree_mont_dauphin_correlation_pva.xyz"
xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/bati_pva.xyz"
#xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/bati_pvaGeomEpip.xyz"
#xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_bati/bati_pva_11_epip_large.xyz"
xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/resultats_bati/bati_pvaGeomEpip_11.xyz"
#xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/metriques/bati_pva_correl-v2_11.xyz"
#xyz = "/home/CHuet2/Documents/monoscopie/chantiers/05_2022/sauvegarde/metriques/bati_pvaGeomEpip_correl-v2_11.xyz"

nb_points_max = 260**2

def open_xyz():
    liste = [] 
    with open(xyz, "r") as f:
        for line in f:
            line_splitted = line.split(" ")
            liste.append([float(line_splitted[0]), float(line_splitted[1]), float(line_splitted[2]), int(line_splitted[3])])
    points_samon = np.array(liste)
    return points_samon


def open_MNS():
    inputds = gdal.Open(mns_lidar_path)
    geotransform = inputds.GetGeoTransform()
    image = inputds.ReadAsArray()
    return image, geotransform


def statistiques_points(points_samon):

    print("Nombre de points calculés : {}".format(points_samon.shape[0]))
    print("Nombre de points non calculés : {}".format(nb_points_max - points_samon.shape[0]))
    for i in range(1, 7):
        condition = np.where(points_samon[:, 3]==i, 1, 0)
        print("Nombre de points calculés par {} pvas : {}, {} %".format(i, np.sum(condition), np.sum(condition)/ nb_points_max * 100))


def get_z_mns_lidar(points_samon, image, geotransform):
    x_min = geotransform[0]
    x_res = geotransform[1]
    y_max = geotransform[3]
    y_res = geotransform[5]

    x_to_colonne = (points_samon[:, 0] - x_min) / x_res
    y_to_ligne = (points_samon[:, 1] - y_max) / y_res
    z_lidar = ndimage.map_coordinates(image, np.vstack([y_to_ligne, x_to_colonne]))
    return z_lidar


def statistiques_on_z(points_samon, z_lidar):
    print("Erreur absolue moyenne en z : {} m".format(np.mean(np.abs(points_samon[:,2] - z_lidar))))
    for i in range(2, 7):
        condition = np.where(points_samon[:, 3]==i, 1, 0)
        nb_points = np.sum(condition)
        moyenne = np.sum(np.abs((points_samon[:,2] - z_lidar)*condition)) / nb_points
        ecart_type = np.sqrt(np.sum(np.abs((points_samon[:,2] - z_lidar)*condition)**2) / nb_points - moyenne**2)
        print("Erreur absolue moyenne en z pour les points avec {} pvas : {} m".format(i, moyenne))
        print("Ecart-type en z pour les points avec {} pvas : {} m".format(i, ecart_type))




points_samon = open_xyz()
statistiques_points(points_samon)
image, geotransform = open_MNS()
z_lidar = get_z_mns_lidar(points_samon, image, geotransform)
print("")
statistiques_on_z(points_samon, z_lidar)
