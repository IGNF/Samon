import numpy as np
from osgeo import gdal, osr

raf_path = "chantiers/49_2020/raf/raf2020.asc"


with open(raf_path, "r") as f:
    dict_compte = {}
    for line in f:
        line_splitted = line.split()
        if str(len(line_splitted)) not in dict_compte:
            dict_compte[str(len(line_splitted))] = 1
        else:
            dict_compte[str(len(line_splitted))] += 1

    print(dict_compte)

    print(16002/381)

    nb_lignes = 381
    nb_colonnes = 10*42+1
    print(nb_lignes)
    print(nb_colonnes)

    array = np.zeros((nb_lignes, nb_colonnes))


with open(raf_path, "r") as f:
    compte_ligne=0
    compte_colonne=0
    for line in f:
        line_splitted = line.split()
        for i in range(0, len(line_splitted)//2):
            array[compte_ligne, compte_colonne] = float(line_splitted[2*i])
            compte_colonne += 1
        if len(line_splitted)==2:
            compte_colonne = 0
            compte_ligne += 1
    print(compte_ligne, compte_colonne)


    print(array)
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create("chantiers/49_2020/raf/raf2020.tif", array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((-5.5, 0.0333333333, 0, 51.5, 0, -0.025))

    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array[:, :])

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(4171)
    outRaster.SetProjection(outSpatialRef.ExportToWkt())
    outband.FlushCache()


