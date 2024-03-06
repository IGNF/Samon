from monoscopie.monoscopie import Monoscopie
import fiona
from shapely.geometry import Point


pva = "chantiers/49_2020/pva"
ortho = "chantiers/49_2020/ortho/ortho.vrt"
mnt =  "chantiers/49_2020/mnt/mnt.vrt"
ta_xml = "chantiers/49_2020/20FD4925_adjust.XML"
resultats = "chantiers/49_2020/resultats_qgis"
points = "chantiers/49_2020/points.shp"
raf = "chantiers/49_2020/raf/raf2020_2154.tif"

monoscopie = Monoscopie(pva, ortho, mnt, ta_xml, resultats, raf)
with fiona.open(points, 'r') as f:
    for point in f:
        point = Point(point["geometry"]["coordinates"][0], point["geometry"]["coordinates"][1])
        monoscopie.run(point)
        print("calculé")