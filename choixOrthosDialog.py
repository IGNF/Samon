from qgis.PyQt.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget, QCheckBox
from .monoscopie.monoscopie import Monoscopie
import os
from .monoscopie.orthoLocale import OrthoLocale
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsRasterLayer, QgsProject, QgsVectorLayer, QgsGeometry, QgsPoint, QgsFeature

class ChoixOrthosDialog(QDialog):
    def __init__(self, projet:Monoscopie):
        QDialog.__init__(self)
        self.projet = projet
        self.layers = []
        self.afficher_images()
        

    def afficher_images(self):
        layout = QHBoxLayout()
        for orthoLocale in self.projet.chantier.ortho_locales:
            orthoLocaleWidget = AffichageOrtho(orthoLocale, self, self.projet)
            layout.addWidget(orthoLocaleWidget)
        self.setLayout(layout)



class AffichageOrtho(QWidget):

    def __init__(self, orthoLocale:OrthoLocale, dialog, projet):
        QWidget.__init__(self)

        self.orthoLocale = orthoLocale
        self.dialog = dialog
        self.projet = projet

        layout = QVBoxLayout()

        canvas = QgsMapCanvas()
        canvas.show()
        canvas.setCanvasColor(Qt.white)
        canvas.enableAntiAliasing(True)
        canvas.setMinimumSize(600, 600)

        canvas_layers = []


        
        # On ajoute le point de corrélation sur l'épipolaire
        point_correlation_layer = QgsVectorLayer("point", "point_correlation", "memory")
        point = QgsGeometry(QgsPoint(orthoLocale.ground_terrain.x, orthoLocale.ground_terrain.y))
        feature = QgsFeature()
        feature.setGeometry(point)
        point_correlation_layer.dataProvider().addFeatures([feature])
        #point_correlation_layer = QgsVectorLayer(chemin_point_correlation, orthoLocale.shot.image+"_point_epip")
        QgsProject.instance().addMapLayer(point_correlation_layer, addToLegend=False)
        self.dialog.layers.append(point_correlation_layer)
        canvas_layers.append(point_correlation_layer)

        if orthoLocale.epip is not None:
            epip_layer = QgsVectorLayer("linestring", "epip", "memory")
            liste_points = [QgsPoint(orthoLocale.epip[0].x, orthoLocale.epip[0].y), QgsPoint(orthoLocale.epip[1].x, orthoLocale.epip[1].y)]
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPolyline(liste_points))
            epip_layer.dataProvider().addFeatures([feature])
            QgsProject.instance().addMapLayer(epip_layer, addToLegend=False)
            self.dialog.layers.append(epip_layer)
            canvas_layers.append(epip_layer)

        #On ajoute l'ortho locale
        raster_layer = QgsRasterLayer(os.path.join(orthoLocale.path_save_pi, orthoLocale.shot.image+".tif"), orthoLocale.shot.image)
        QgsProject.instance().addMapLayer(raster_layer, addToLegend=False)#Indispensable pour que la couche apparaisse dans le canvas
        self.dialog.layers.append(raster_layer)
        canvas_layers.append(raster_layer)

        #On ajoute les couches au canvas et on centre le canvas sur l'ortho'
        canvas.setExtent(raster_layer.extent())
        canvas.setLayers(canvas_layers)

        layout.addWidget(canvas)

        # On ajoute une checkbox qui est cochée si l'ortho a été considérée comme valide pour le calcul
        self.checkbox = QCheckBox("{} : {}".format(orthoLocale.shot.image, int(orthoLocale.correlation*1000)/1000))
        if orthoLocale.calcul_valide:
            self.checkbox.setChecked(True)

        # On ajoute un événement qui relance le calcul dès qu'une checkbox est cochée ou décochée
        self.checkbox.stateChanged.connect(self.state_changed)
        
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def state_changed(self):
        self.orthoLocale.calcul_valide = self.checkbox.isChecked()
        
        self.dialog.done(1)
        
        