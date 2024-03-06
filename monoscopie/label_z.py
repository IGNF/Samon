from qgis.core import QgsVectorLayer, QgsCoordinateTransformContext, QgsCoordinateReferenceSystem, QgsField, QgsFeature, QgsGeometry, QgsPoint
from qgis.core import QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling
from qgis.PyQt.QtCore import QVariant 
from qgis.PyQt.QtGui import QColor, QFont
from .tool import print_log
from .infosResultats import InfosResultats

class GeometryLabelZ():

    def __init__(self, context:QgsCoordinateTransformContext) -> None:
        #type_geom: point, linestring, polygon
        self.context:QgsCoordinateTransformContext = context
        self.id_3d:int = 0
        self.geometry_3d:QgsGeometry = QgsGeometry
        self.liste_id = []
        self.liste_z = []
        self.create_shapefile_3D()


    def create_shapefile_3D(self)->None:
        self.vl_3D = QgsVectorLayer("point", "altitude", "memory")
        self.vl_3D.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(2154))
        self.vl_3D.startEditing()
        self.vl_3D.dataProvider().addAttributes([
            QgsField("id", QVariant.Int),
            QgsField("z", QVariant.Double),
            QgsField("dz", QVariant.Double),
            QgsField("affichage", QVariant.String),
            QgsField("dz_lidar", QVariant.Double),
        ])
        self.vl_3D.updateFields()
        settings = QgsPalLayerSettings()
        format = QgsTextFormat()
        format.setFont(QFont('Arial', 8))
        format.setColor(QColor('Black'))
        settings.setFormat(format)
        settings.fieldName = "affichage"
        settings.isExpression = True
        labels = QgsVectorLayerSimpleLabeling(settings)
        self.vl_3D.setLabelsEnabled(True)
        self.vl_3D.setLabeling(labels)

    
    def supprimer_feature(self, id_alti)->None:
        self.vl_3D.startEditing()
        self.vl_3D.deleteFeature(id_alti)
        self.vl_3D.commitChanges()
 

    def ajouter_point(self, infoResultats:InfosResultats)->int:

        point = QgsPoint(infoResultats.point3d[0], infoResultats.point3d[1], infoResultats.point3d[2])
        geometry_3d = QgsGeometry(point)
        f3d = QgsFeature()
        z = float(int(infoResultats.altitude*100)/100)
        dz = float(int(infoResultats.ecart_alti*100)/100)
        if infoResultats.ecart_z_lidar is not None:
            dz_lidar = float(int(infoResultats.ecart_z_lidar*100)/100)
            f3d.setAttributes([infoResultats.id, z, dz, "{} ; {} ; {}".format(z, dz, dz_lidar), dz_lidar])
        else:
            f3d.setAttributes([infoResultats.id, z, dz, "{} ; {}".format(z, dz)])
        f3d.setGeometry(geometry_3d)
        (_, new_feature) = self.vl_3D.dataProvider().addFeatures([f3d])
        self.vl_3D.triggerRepaint()
        return new_feature[0].id()
        