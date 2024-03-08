import argparse
from lxml import etree
import os
from osgeo import ogr, osr

def findEPSG(root):
    projection = root.find(".//projection").text.strip()
    dictionnaire = {
        "UTM_WGS84_f01sud":32701,
        "UTM_WGS84_f05sud":32705,
        "UTM_WGS84_f06sud":32706,
        "UTM_WGS84_f07sud":32707,
        "UTM_WGS84_f12nord":32612,
        "UTM_WGS84_f20nord":32620,
        "UTM_WGS84_f21nord":32621,
        "UTM_WGS84_f22nord":32622,
        "UTM_WGS84_f30nord":32630,
        "UTM_WGS84_f31nord":32631,
        "UTM_WGS84_f32nord":32632,
        "UTM_WGS84_f33nord":32633,
        "UTM_WGS84_f34nord":32634,
        "UTM_WGS84_f36nord":32636,
        "UTM_WGS84_f38sud":32738,
        "UTM_WGS84_f39sud":32739,
        "UTM_WGS84_f40sud":32740,
        "UTM_WGS84_f42sud":32742,
        "UTM_WGS84_f43sud":32743,
        "UTM_WGS84_f58sud":32758,
        "Lambert93":2154,
    }
    EPSG = dictionnaire[projection]
    print("L'EPSG du chantier est {}".format(EPSG))

    return EPSG


def lecture_xml(path, selected_images=None):
    #Récupère pour chaque cliché le nom de l'image et l'emprise donnée par le fichier xml
    tree = etree.parse(path)
    root = tree.getroot()

    images = []
    
    for cliche in root.getiterator("cliche"):
        image = {}
        image_nom = cliche.find("image").text.strip()
        if selected_images is None or image_nom in selected_images:
            image["nom"] = image_nom
            polygon2d = cliche.find("polygon2d")
            x = polygon2d.findall("x")
            y = polygon2d.findall("y")
            x = [float(i.text) for i in x]
            y = [float(i.text) for i in y]
        
            image["x"] = x
            image["y"] = y
            images.append(image)

    EPSG = findEPSG(root)

    return images, EPSG

def save_EPSG(analyse_plan_vol_path, EPSG):
    with open(os.path.join(analyse_plan_vol_path, "EPSG.txt"), "w") as f:
        f.write(str(EPSG))

def save_shapefile(images, path, EPSG):
    #Sauvegarde les emprises dans un fichier shapefile

    driver = ogr.GetDriverByName("ESRI Shapefile")

    ds = driver.CreateDataSource(path)
    srs =  osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    layer = ds.CreateLayer("line", srs, ogr.wkbPolygon)

    nameField = ogr.FieldDefn("nom", ogr.OFTString)
    layer.CreateField(nameField)

    featureDefn = layer.GetLayerDefn()

    xmin = 1e15
    ymin = 1e15
    xmax = -1e15
    ymax = -1e15

    for image in images:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i in range(len(image["x"])):
            ring.AddPoint(image["x"][i], image["y"][i])
            xmin = min(xmin, image["x"][i])
            ymin = min(ymin, image["y"][i])
            xmax = max(xmax, image["x"][i])
            ymax = max(ymax, image["y"][i])
        ring.AddPoint(image["x"][0], image["y"][0])

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
                
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("nom", image["nom"])
        layer.CreateFeature(feature)
        feature = None
    ds = None
    return [xmin, ymin, xmax, ymax]

def visualisation_chantier(ta_path):
    chantier_path = os.path.dirname(ta_path)
    analyse_plan_vol_path = os.path.join(chantier_path, "Analyse_Plan_Vol")
    os.makedirs(analyse_plan_vol_path, exist_ok=True)
    images, EPSG = lecture_xml(ta_path)
    metadata_path = os.path.join(chantier_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)
    save_EPSG(metadata_path, EPSG)
    emprise_path = os.path.join(analyse_plan_vol_path, "chantier.shp")
    save_shapefile(images, emprise_path, EPSG)
    return emprise_path
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Visualisation de la position approximative des chantiers")
    parser.add_argument('--ta', help='Fichier xml du chantier')
    args = parser.parse_args()
    ta_path = args.ta

    visualisation_chantier(ta_path)