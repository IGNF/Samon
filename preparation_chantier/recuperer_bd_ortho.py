import geopandas as gpd
import math
from shapely.geometry import Polygon
import os
import argparse
import shutil
from osgeo import gdal
from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtWidgets import QProgressBar
from qgis.utils import iface



EPSG_DEP = {
    "32620": "971", # 972,
    "32738": "976",
    "32740": "974",
    "32622": "973",
}


EPSG_WGS84_RGAF = {
    "32620": 5490,
    "32738": 4471,
    "32740": 2975,
    "32622": 2972,
    "2154": 2154
}

EPSG_NOM = {
    "32620": "RRAFUTM20",
    "32738": "RGM04UTM38S",
    "32740": "RGR92UTM40S",
    "32622": "RGFG95UTM22",
    "2154": "RGF93LAMB93"
}


def load_bbox(metadata):
    bbox = []
    with open(os.path.join(metadata, "bbox.txt"), "r") as f:
        for line in f:
            bbox.append(float(line.strip()))
    
    tile_factor = 1
    bbox[0] = math.floor(bbox[0]/(tile_factor*1000)) * tile_factor
    bbox[1] = math.floor(bbox[1]/(tile_factor*1000)) * tile_factor
    bbox[2] = math.ceil(bbox[2]/(tile_factor*1000)) * tile_factor
    bbox[3] = math.ceil(bbox[3]/(tile_factor*1000)) * tile_factor
    return bbox

def getEPSG(metadata):
    with open(os.path.join(metadata, "EPSG.txt"), "r") as f:
        for line in f:
            return int(line)

def get_emprise_tiles(bbox, EPSG):

    #On divise l'emprise en tuiles de 1 km de côté
    tmp_list = []
    compte = 0
    for e in range(bbox[0], bbox[2], 1):
        for n in range(bbox[1], bbox[3], 1):
            geometry = Polygon([(e*1000, n*1000), ((e+1)*1000, n*1000), ((e+1)*1000, (n+1)*1000), (e*1000, (n+1)*1000)])
            
            # Si c'est en France métropolitaine, alors on détermine le département avec une jointure sur les départements de la BDAdmin
            if EPSG == 2154:
                tmp_list.append({
                    'geometry' : geometry,
                    'id': compte,
                    "name": "{}_{}".format(e, n)
                })
            else:# Si c'est dans les DOM-TOM, alors on détermine le département à partir de l'EPSG
                tmp_list.append({
                    'geometry' : geometry,
                    'id': compte,
                    "name": "{}_{}".format(e, n),
                    "INSEE_DEP":EPSG_DEP[str(EPSG)]
                })
            compte += 1
    
    # On met les tuiles dans l'EPSG des données de store-ref
    # En effet, dans les métadonnées Hiatus, les données sont en UTM WGS84
    # Dans store-ref, elles sont en UTM projection locale
    emprise_tiles = gpd.GeoDataFrame(tmp_list).set_crs(epsg=EPSG).to_crs(epsg=EPSG_WGS84_RGAF[str(EPSG)])

    if EPSG == 2154:
        #On charge le shapefile contenant les départements
        plugin_path = os.path.dirname(os.path.realpath(__file__))
        departements = gpd.read_file(os.path.join(plugin_path, "ADMIN-EXPRESS", "DEPARTEMENT.shp"))

        #On effectue une jointure entre les tuiles et les départements
        emprise_tiles_join = gpd.sjoin(emprise_tiles, departements)

        return emprise_tiles_join

    else:
        return emprise_tiles





def get_dalle_MNT(bbox, chantier, store_ref):
    n_max = int(math.ceil(bbox[3]/10)*10000)
    e_min = int(math.floor(bbox[0]/10)*10000)
    for e in range(e_min, int(bbox[2])*1000, 10000):
        for n in range(n_max, int(bbox[1])*1000, -10000):

            chemin = os.path.join(store_ref, "modeles-numeriques-3D", "RGEAlti", "2024", "RGEALTI_MNT_1M00_ASC_RGF93LAMB93_FXX")
            nom_fichier = "{}-{}.asc.gz".format(e, n)
            chemin_fichier = os.path.join(chemin, nom_fichier)
            if os.path.exists(chemin_fichier):
                if not os.path.exists(nom_fichier):
                    shutil.copy(chemin_fichier, os.path.join(chantier, "mnt", nom_fichier))
            else:
                print("Impossible de trouver : {}".format(chemin_fichier))


def get_dalles(chantier, emprise_tiles, EPSG, annee, store_ref):

    progressMessageBar = iface.messageBar().createMessage("Chargement des dalles de BD Ortho...")
    progress = QProgressBar()
    progress.setMaximum(emprise_tiles.shape[0])
    progressMessageBar.layout().addWidget(progress)
    iface.messageBar().pushWidget(progressMessageBar, Qgis.Info)
    
    #On parcourt toutes les tuiles de l'emprise
    for i, tile in enumerate(emprise_tiles.iterfeatures()):
        departement_tile = tile["properties"]["INSEE_DEP"]

        #On récupère les coordonnées de la tuile
        e_min = int(tile["geometry"]["coordinates"][0][0][0] / 1000)
        n_min = int(tile["geometry"]["coordinates"][0][0][1] / 1000)

        if e_min < 1000:
            e_min = "0" + str(e_min)
        else:
            e_min = str(e_min)

        if n_min < 1000:
            n_min = "0" + str(n_min)
        else:
            n_min = str(n_min)

        get_dalle_ortho(chantier, e_min, n_min, annee, departement_tile, tile, EPSG, store_ref)
        progress.setValue(i)
    iface.messageBar().clearWidgets()
        

def get_annee(path, annee_souhaitee):
    annees_possibles = os.listdir(path)
    if annee_souhaitee in annees_possibles:
        return annee_souhaitee
    else:
        annee_souhaitee_int = int(annee_souhaitee)
        difference_min = 1e15
        annee_min = 0
        for a in annees_possibles:
            difference = abs(annee_souhaitee_int - int(a))
            if difference < difference_min:
                annee_min = a
                difference_min = difference
        print("L'année {} n'a pas été trouvée, elle a été remplacée par {}".format(annee_souhaitee, annee_min))
        return annee_min



def get_dalle_ortho(chantier, e_min, n_min, annee, departement, tile, EPSG, store_ref):
    if len(departement) <= 2:
        departement = "D0" + departement
    else:
        departement = "D" + departement


    annee = get_annee(os.path.join(store_ref, "ortho-images", "Ortho", departement), annee)
    dossier_departement = "BDORTHO_RVB-0M20_JP2-E100_{}_{}_{}".format(EPSG_NOM[str(EPSG)], departement, annee)
    
    chemin_ortho = os.path.join(store_ref, "ortho-images", "Ortho", departement, str(annee), dossier_departement)

    if os.path.exists(chemin_ortho):
        exemple_fichier = os.listdir(chemin_ortho)[0][:-4].split("-")
        exemple_fichier[2] = e_min
        exemple_fichier[3] = str(int(n_min) +1)
        fichier = "-".join(exemple_fichier) + ".jp2"
        if os.path.exists(os.path.join(chemin_ortho, fichier)):

            chemin_image_local = os.path.join(chantier, "ortho", "ORTHO_{}.tif".format(tile["properties"]["name"]))
            ds = gdal.Open(os.path.join(chemin_ortho, fichier))
            ds = gdal.Translate(chemin_image_local, ds)
    else:
        print("Le répertoire {} n'existe pas".format(chemin_ortho))
        


def recuperer_bd_ortho(annee, chantier, store_ref):

    metadata = os.path.join(chantier, "metadata")

    if not os.path.exists(os.path.join(chantier, "ortho")):
        os.makedirs(os.path.join(chantier, "ortho"))

    if not os.path.exists(os.path.join(chantier, "mnt")):
        os.makedirs(os.path.join(chantier, "mnt"))


    #On récupère l'emprise du chantier, arrondie au kilomètre
    bbox = load_bbox(metadata)

    #On récupère l'EPSG du chantier
    EPSG = getEPSG(metadata)

    #On divise le chantier en dalles de 1 km de côté
    emprise_tiles = get_emprise_tiles(bbox, EPSG)
    emprise_tiles.to_file(os.path.join(chantier, "emprise.shp"))

    get_dalles(chantier, emprise_tiles, EPSG, annee, store_ref)
    get_dalle_MNT(bbox, chantier, store_ref)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Vérification qu'il n'y a pas d'images isolées sur le chantier")
    parser.add_argument('--annee', help='Année pour la BD Ortho')
    parser.add_argument('--chantier', help='Répertoire du chantier')
    args = parser.parse_args()

    recuperer_bd_ortho(args.annee, args.chantier, os.path.join("/media", "store-ref"))