from qgis.core import QgsVectorLayer, QgsCoordinateTransformContext, QgsCoordinateReferenceSystem, QgsField, QgsVectorFileWriter, QgsFeature, QgsGeometry, QgsLineString, QgsPoint, QgsMessageLog, QgsPolygon
from qgis.PyQt.QtCore import QVariant
import os
from .tool import print_log
from .infosResultats import InfosResultats
from typing import List



class LayerGeometry:

    def __init__(self, type_geom:str, nom_couche_temp:str, context:QgsCoordinateTransformContext, sortie:str) -> None:
        self.type_geom:str = type_geom
        self.nom_couche_temp:str = nom_couche_temp
        self.context:QgsCoordinateTransformContext = context
        self.sortie:str = sortie
        self.liste_objectGeometry_2D:List[ObjectGeometry] = []
        self.liste_objectGeometry_3D:List[ObjectGeometry] = []
        self.create_shapefile_2D()
        self.create_shapefile_3D()

    def create_shapefile_2D(self)->None:
        self.vl_2D = QgsVectorLayer(self.type_geom, self.nom_couche_temp+"_2D", "memory")
        self.vl_2D.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(2154))
        self.vl_2D.dataProvider().addAttributes([QgsField("id", QVariant.Int)])
        self.vl_2D.updateFields()


    def create_shapefile_3D(self)->None:
        self.vl_3D = QgsVectorLayer(self.type_geom, self.nom_couche_temp+"_3D", "memory")
        self.vl_3D.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(2154))
        self.vl_3D.dataProvider().addAttributes([
            QgsField("id", QVariant.Int),
            QgsField("z", QVariant.Double),
            QgsField("nb_images", QVariant.Int),
            QgsField("residus", QVariant.Double)
        ])
        self.vl_3D.updateFields()


    def sauvegarde_shapefile(self)->None:
        print_log("Début de la sauvegarde de la géométrie")
        # On sauvegarde les coordonnées 2D du point cliqué
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "ESRI Shapefile"
        options.layerName = 'points_2D'
        QgsVectorFileWriter.writeAsVectorFormatV3(
            self.vl_2D,
            os.path.join(self.sortie, self.nom_couche_temp+"_2D.shp"),
            self.context,
            options
        )

        # On sauvegarde les coordonnées 3D du point après calcul
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "ESRI Shapefile"
        options.layerName = 'points_3D'
        QgsVectorFileWriter.writeAsVectorFormatV3(
            self.vl_3D,
            os.path.join(self.sortie, self.nom_couche_temp+"_3D.shp"),
            self.context,
            options
        )

    def completer_figure(self):
        if len(self.liste_objectGeometry_2D) >= 1:
            current_geom_2D = self.liste_objectGeometry_2D[-1]
            current_geom_3D = self.liste_objectGeometry_3D[-1]
            current_geom_2D.indiquer_complet(True)
            current_geom_3D.indiquer_complet(True)
        
        
    def retirer_derniere_geometrie_2D(self):
        #Parce que pop() et del font crasher QGis...
        l = self.liste_objectGeometry_2D.copy()
        self.liste_objectGeometry_2D = []
        for i in range(len(l)-1):
            self.liste_objectGeometry_2D.append(l[i])


    def retirer_derniere_geometrie_3D(self):
        l = self.liste_objectGeometry_3D.copy()
        self.liste_objectGeometry_3D = []
        for i in range(len(l)-1):
            self.liste_objectGeometry_3D.append(l[i])

    def retirer_point(self)->int:
        id_alti = None
        # On retire un point seulement s'il y a au moins une géométrie dans la liste
        if len(self.liste_objectGeometry_2D) >= 1:
            # On récupère la dernière géométrie
            current_geom_2D = self.liste_objectGeometry_2D[-1]
            current_geom_3D = self.liste_objectGeometry_3D[-1]
            
            # S'il y a au moins un point
            if len(current_geom_2D.liste_points) >= 1 and not current_geom_2D.complet:

                # On retire le dernier point 
                current_geom_2D.retirer_point()
                id_alti = current_geom_3D.retirer_point()



                # S'il y a au moins un point dans la géométrie
                if len(current_geom_2D.liste_points) >= 1:
                    # On met à jour la géométrie
                    current_geom_2D.creer_geometrie()
                    current_geom_3D.creer_geometrie()

                    # On écrit la géométrie
                    self.ecrire_geometries(current_geom_2D, current_geom_3D, change=True)
        return id_alti

    



    def ajouter_point(self, infoResultats:InfosResultats, id_alti:int)->None:

        if len(self.liste_objectGeometry_2D) >= 1:
            current_geom_2D = self.liste_objectGeometry_2D[-1]
            current_geom_3D = self.liste_objectGeometry_3D[-1]
        else:
            self.liste_objectGeometry_2D.append(ObjectGeometry(self.type_geom))
            self.liste_objectGeometry_3D.append(ObjectGeometry(self.type_geom, pointZ = True))
            current_geom_2D = self.liste_objectGeometry_2D[-1]
            current_geom_3D = self.liste_objectGeometry_3D[-1]

        if current_geom_2D.complet:
            self.liste_objectGeometry_2D.append(ObjectGeometry(self.type_geom))
            self.liste_objectGeometry_3D.append(ObjectGeometry(self.type_geom, pointZ = True))
            current_geom_2D = self.liste_objectGeometry_2D[-1]
            current_geom_3D = self.liste_objectGeometry_3D[-1]



        current_geom_2D.ajouter_point(infoResultats, id_alti)
        current_geom_3D.ajouter_point(infoResultats, id_alti)

        self.ecrire_geometries(current_geom_2D, current_geom_3D)

 

    def ecrire_geometries(self, current_geom_2D, current_geom_3D, change=False)->None:
        

        if len(current_geom_2D.liste_points) >= 2 or change:
            self.vl_2D.dataProvider().changeGeometryValues({ current_geom_2D.id_modification : current_geom_2D.geometry_2d })
            self.vl_3D.dataProvider().changeGeometryValues({ current_geom_3D.id_modification : current_geom_3D.geometry_3d })
        else:
            geom_2d_copy = current_geom_2D.geometry_2d
            geom_3d_copy = current_geom_3D.geometry_3d
            current_geom_2D.feature.setAttributes(current_geom_2D.getAttributes())
            current_geom_2D.feature.setGeometry(geom_2d_copy)
            current_geom_3D.feature.setAttributes(current_geom_3D.getAttributes())
            current_geom_3D.feature.setGeometry(geom_3d_copy)

            (_, newFeatures) = self.vl_2D.dataProvider().addFeatures([current_geom_2D.feature])
            current_geom_2D.id_modification = newFeatures[0].id()

            (_, newFeatures) = self.vl_3D.dataProvider().addFeatures([current_geom_3D.feature])
            current_geom_3D.id_modification = newFeatures[0].id()
        self.vl_2D.triggerRepaint()
        self.vl_3D.triggerRepaint()

        



class ObjectGeometry:

    def __init__(self, type_geom, pointZ=False) -> None:
        self.type_geom = type_geom
        self.geometry = None
        self.liste_id = []
        self.liste_z = []
        self.liste_nb_images = []
        self.liste_residus = []
        self.liste_points = []
        self.pointZ = pointZ
        self.id_modification = None
        self.feature = QgsFeature()
        self.complet = False
        self.liste_id_alti = []

    def __str__(self) -> str:
        return "nb_points : {}, liste : {}".format(len(self.liste_points), self.liste_points)

    def ajouter_point(self, infoResultats:InfosResultats, id_alti:int)->None:
        # On ajoute le point dans la liste
        if self.pointZ:
            self.liste_points.append(QgsPoint(infoResultats.point3d[0], infoResultats.point3d[1], infoResultats.point3d[2]))
            self.liste_z.append(infoResultats.altitude)
            self.liste_nb_images.append(infoResultats.nb_images)
            self.liste_residus.append(infoResultats.residu)
            self.liste_id_alti.append(id_alti)
        else:
            self.liste_points.append(QgsPoint(infoResultats.point2d.x, infoResultats.point2d.y))
        
        self.liste_id.append(infoResultats.id)
        self.creer_geometrie()


    def retirer_point(self)->int:
        del(self.liste_points[-1])
        del(self.liste_id[-1])
        id_alti = None
        if self.pointZ:
            del(self.liste_z[-1])
            del(self.liste_nb_images[-1])
            del(self.liste_residus[-1])
            id_alti = self.liste_id_alti.pop()
        
        self.indiquer_complet(False)

        return id_alti


    
    def creer_geometrie(self):
        # On crée la nouvelle géométrie
        if self.type_geom == "point":
            point = self.liste_points.copy()[0]
            self.geometry_2d = QgsGeometry(point)
            self.geometry_3d = QgsGeometry(point)
            self.indiquer_complet(True)
        elif self.type_geom == "linestring":
            self.geometry_2d = QgsGeometry.fromPolyline(self.liste_points)
            self.geometry_3d = QgsGeometry.fromPolyline(self.liste_points)
        elif self.type_geom == "polygon":
            points_2d_poly = self.liste_points.copy()
            points_2d_poly.append(self.liste_points[0])
            points_3d_poly = self.liste_points.copy()
            points_3d_poly.append(self.liste_points[0])
            self.geometry_2d = QgsGeometry(QgsPolygon(QgsLineString(points_2d_poly)))
            self.geometry_3d = QgsGeometry(QgsPolygon(QgsLineString(points_3d_poly)))

    def indiquer_complet(self, booleen):
        self.complet = booleen
        


    def getAttributes(self):
        if self.pointZ:
            return [self.liste_id[0], float(self.liste_z[0]), self.liste_nb_images[0], float(self.liste_residus[0])]
        else:
            return [self.liste_id[0]]
        