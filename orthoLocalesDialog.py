from qgis.PyQt.QtWidgets import QDialog, QHBoxLayout
from qgis.gui import QgsMapCanvas, QgsMapToolEmitPoint
from qgis.core import QgsRasterLayer, QgsProject
from .monoscopie.monoscopie import Monoscopie
import os
from shapely.geometry import Point
from .monoscopie.orthoLocale import OrthoLocale
from qgis.PyQt.QtCore import Qt


class OrthoLocalesDialog(QDialog):
    def __init__(self, projet:Monoscopie, samon):
        QDialog.__init__(self)
        self.projet = projet
        self.samon = samon
        self.pointTools = []
        self.raster_layers:QgsRasterLayer = []
        
        self.afficher_images()
        

    def afficher_images(self):
        layout = QHBoxLayout()
        for orthoLocale in self.projet.chantier.ortho_locales:

            #On crée le canvas pour chaque raster
            canvas = QgsMapCanvas()
            canvas.show()
            canvas.setCanvasColor(Qt.white)
            canvas.enableAntiAliasing(True)
            canvas.setMinimumSize(600, 600)

            #On crée la couche raster
            raster_layer = QgsRasterLayer(os.path.join(orthoLocale.path_save_pi, orthoLocale.shot.image+".tif"), orthoLocale.shot.image)
            QgsProject.instance().addMapLayer(raster_layer, addToLegend=False)#Indispensable pour que la couche apparaisse dans le canvas
            self.raster_layers.append(raster_layer)
            
            #On ajoute la couche au canvas et on centre le canvas sur la couche
            canvas.setExtent(raster_layer.extent())
            canvas.setLayers([raster_layer])
            
            layout.addWidget(canvas)

            #On ajoute l'événement clic sur le canvas
            pointTool = QgsMapToolEmitPoint(canvas)
            pointTool.canvasClicked.connect(lambda e, m, orthoLocale=orthoLocale : self.img_click(e, m, orthoLocale))
            canvas.setMapTool(pointTool)
            #Il faut conserver les pointTool sinon rien ne se passe quand on clique sur les canvas
            self.pointTools.append(pointTool)

        self.setLayout(layout)


    
    def img_click(self, event, mouseButton, orthoLocale:OrthoLocale):
        centre = Point(event.x(), event.y())
        self.done(1)

        #On retire les couches (sinon, on les voit lorsque l'on relance une deuxième fois le plugin dans les champs ortho, mnt et raf du formulaire)
        for raster_layer in self.raster_layers:
            QgsProject.instance().removeMapLayer(raster_layer)
        
        #On calcule le point
        self.projet.run(centre, orthoLocaleMaitresse=orthoLocale)
        
        #On ajoute le point dans le couches shapefile
        self.samon.clic_sur_ortho_non_maitresse()
