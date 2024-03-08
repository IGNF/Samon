import argparse
import os
import geopandas as gpd
from .prepa_chantier_1 import lecture_xml, save_shapefile
from osgeo import gdal
from .recuperer_bd_ortho import recuperer_bd_ortho
import gzip
import shutil
from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtWidgets import QProgressBar
from qgis.utils import iface



def get_selected_images(selection_path):
    images = []
    images_gdf = gpd.read_file(selection_path)
    for image in images_gdf.iterfeatures():
        images.append(image["properties"]["nom"])
    return images


def save_bbox(analyse_plan_vol_path, bbox):
    with open(os.path.join(analyse_plan_vol_path, "bbox.txt"), "w") as f:
        for i in bbox:
            f.write("{}\n".format(i))


def copy_images(selected_images, pvas_path, images_path):
    progressMessageBar = iface.messageBar().createMessage("Chargement des images orientées...")
    progress = QProgressBar()
    progress.setMaximum(len(selected_images))
    progressMessageBar.layout().addWidget(progress)
    iface.messageBar().pushWidget(progressMessageBar, Qgis.Info)
    for i, selected_image in enumerate(selected_images):
        image_store_ref_path = os.path.join(images_path, "{}.jp2".format(selected_image))
        if os.path.exists(image_store_ref_path):
            ds = gdal.Open(image_store_ref_path)
            chantier_images_path = os.path.join(pvas_path, "{}.tif".format(selected_image))
            ds = gdal.Translate(chantier_images_path, ds)
        else:
            QgsMessageLog.logMessage("Image not found : {}".format(image_store_ref_path), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        progress.setValue(i)
    iface.messageBar().clearWidgets()

def unzip_mnt(path_mnt):
    tiles = [i for i in os.listdir(path_mnt) if i[-3:]==".gz"]
    progressMessageBar = iface.messageBar().createMessage("Dézippage du MNT...")
    progress = QProgressBar()
    progress.setMaximum(len(tiles))
    progressMessageBar.layout().addWidget(progress)
    iface.messageBar().pushWidget(progressMessageBar, Qgis.Info)
    
    for i, tile in enumerate(tiles):
        tile_asc = tile[:-3]
        with gzip.open(os.path.join(path_mnt, tile), 'rb') as f_in:
            with open(os.path.join(path_mnt, tile_asc), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(path_mnt, tile))
        progress.setValue(i)

    commande = "gdalbuildvrt {} {}".format(os.path.join(path_mnt, "mnt.vrt"), os.path.join(path_mnt, "*.asc"))
    os.system(commande)
    iface.messageBar().clearWidgets()
        


def prepa_chantier_2(ta_path, selection_path, images_path, storeref_path):
    chantier_path = os.path.dirname(ta_path)
    
    QgsMessageLog.logMessage("Lecture du fichier {}".format(selection_path), tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On ouvre le fichier shapefile filtré par les utilisateurs
    selected_images = get_selected_images(selection_path)
    
    QgsMessageLog.logMessage("Lecture du ta", tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On ouvre le ta
    images, EPSG = lecture_xml(ta_path, selected_images)

    QgsMessageLog.logMessage("Création de chantier_selection.shp", tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On sauvegarde les emprises au sol des images conservées
    analyse_plan_vol_path = os.path.join(chantier_path, "Analyse_Plan_Vol")
    bbox = save_shapefile(images, os.path.join(analyse_plan_vol_path, "chantier_selection.shp"), EPSG)
    
    # On sauvegarde la bounding box du chantier
    metadata_path = os.path.join(chantier_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)
    save_bbox(metadata_path, bbox)

    QgsMessageLog.logMessage("Copie des images orientées", tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On récupère les images orientées
    pvas_path = os.path.join(chantier_path, "pvas")
    os.makedirs(pvas_path, exist_ok=True)
    copy_images(selected_images, pvas_path, images_path)
    
    QgsMessageLog.logMessage("Récupération de la BD Ortho et du MNT", tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On récupère la BD Ortho et le MNT
    metadata_path = os.path.join(chantier_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)
    dirname = os.path.basename(os.path.normpath(images_path))
    annee = dirname.split("_")[0]
    recuperer_bd_ortho(annee, chantier_path, storeref_path)
    commande = "gdalbuildvrt {} {}".format(os.path.join(chantier_path, "ortho", "ortho.vrt"), os.path.join(chantier_path, "ortho", "OR*.tif"))
    os.system(commande)

    QgsMessageLog.logMessage("Dézippage du MNT", tag="Infos Samon", level=Qgis.MessageLevel.Info)
    # On dézippe le MNT
    unzip_mnt(os.path.join(chantier_path, "mnt"))











if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Visualisation de la position approximative des chantiers")
    parser.add_argument('--ta', help='Fichier xml du chantier')
    parser.add_argument('--selection', help='Fichier shapefile contenant les emprises au sol des images à conserver')
    parser.add_argument('--images', help='Répertoire contenant les images orientées')
    parser.add_argument('--storeref', help='Chemin vers le store-ref')
    args = parser.parse_args()
    ta_path = args.ta
    selection_path = args.selection
    images_path = args.images
    storeref_path = args.storeref
    

    prepa_chantier_2(ta_path, selection_path, images_path, storeref_path)