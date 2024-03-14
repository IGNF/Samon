# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Samon
                                 A QGIS plugin
 Saisie monoscopique sur la BD Ortho
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-09-14
        git sha              : $Format:%H$
        copyright            : (C) 2023 by IGN
        email                : celestin.huet@ign.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QShortcut

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .samon_dialog import SamonDialog
import os.path
from .monoscopie.monoscopie import Monoscopie
from qgis.gui import QgsMapToolEmitPoint
from shapely.geometry import Point
from .monoscopie.tool import print_log
from qgis.core import QgsProject, QgsMessageLog, Qgis, QgsVectorLayer, QgsRasterLayer
from .monoscopie.geometry import LayerGeometry
import numpy as np
from.monoscopie.label_z import GeometryLabelZ
from .orthoLocalesDialog import OrthoLocalesDialog
from .choixOrthosDialog import ChoixOrthosDialog
from .help import HelpDialog
from .monoscopie.lidar import MNSLidar
from .monoscopie.infosResultats import InfosResultats
import time
from .preparation_chantier.prepa_chantier_1 import visualisation_chantier
from .preparation_chantier.prepa_chantier_2 import prepa_chantier_2
from qgis.utils import iface


class Samon:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'Samon_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Samon')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

        self.shortcut_point = None
        self.shortcut_polyligne = None
        self.shortcut_polygone = None

        self.geom_point = None
        self.geom_linestring = None
        self.geom_polygon = None
        self.shortcut_delete = None
        self.shortcut_meme_bande = None
        self.shortcut_saisie_pva = None

        self.mode = "point"
        self.last_z = []
        self.last_point = []

        self.projet:Monoscopie = None

        self.meme_bande = False

        self.mns_lidar_path = None
        self.MNSLidar:MNSLidar = None

        if self.mns_lidar_path:
            self.MNSLidar = MNSLidar(self.mns_lidar_path)

        self.somme_temps = 0
        self.nb_points = -1


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Samon', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/samon/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Samon'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Samon'),
                action)
            self.iface.removeToolBarIcon(action)

        if self.shortcut_point is not None:
            self.shortcut_point.activated.disconnect()
        if self.shortcut_polyligne is not None:
            self.shortcut_polyligne.activated.disconnect()
        if self.shortcut_polygone is not None:
            self.shortcut_polygone.activated.disconnect()
        if self.shortcut_delete is not None:
            self.shortcut_delete.activated.disconnect()
        if self.shortcut_meme_bande is not None:
            self.shortcut_meme_bande.activated.disconnect()
        if self.shortcut_saisie_pva is not None:
            self.shortcut_saisie_pva.activated.disconnect()


    def select_file(self, widget, text, extensions):
        filename = QFileDialog.getOpenFileName(
            self.dlg, text, "/home", extensions)
        widget.setText(filename[0])

    def select_directory(self, widget, text):
        filename = QFileDialog.getExistingDirectory(
            self.dlg, text)
        widget.setText(filename)

    
    def ajouter_point(self, infosResultats:InfosResultats):

        if self.MNSLidar is not None:
            ecart_z_lidar = self.MNSLidar.get_value(infosResultats.point3d[0], infosResultats.point3d[1])
            if ecart_z_lidar is not None:
                infosResultats.set_ecart_z_lidar(ecart_z_lidar)

        id_alti = self.geom_altitude.ajouter_point(infosResultats)
        if self.mode == "point":
            self.geom_point.ajouter_point(infosResultats, id_alti)
        if self.mode == "linestring":
            self.geom_linestring.ajouter_point(infosResultats, id_alti)
        if self.mode == "polygon":
            self.geom_polygon.ajouter_point(infosResultats, id_alti)
        

    def sauvegarder_shapefile(self):
        print_log("Sauvegarde de la géométrie")
        if self.mode == "point":
            self.geom_point.sauvegarde_shapefile()
        if self.mode == "linestring":
            self.geom_linestring.sauvegarde_shapefile()
        if self.mode == "polygon":
            self.geom_polygon.sauvegarde_shapefile()

    def geom_initialisation(self):
        if self.mode == "point":
            self.geom_point.completer_figure()
        if self.mode == "linestring":
            self.geom_linestring.completer_figure()
        if self.mode == "polygon":
            self.geom_polygon.completer_figure()
        #self.geom_altitude.initialisation()

    def geom_supprimer_dernier_point(self):
        id_alti = None
        if self.mode == "point":
            id_alti = self.geom_point.retirer_point()
            
        elif self.mode == "linestring":
            id_alti = self.geom_linestring.retirer_point()
        elif self.mode == "polygon":
            id_alti = self.geom_polygon.retirer_point()
        if id_alti:
            self.geom_altitude.supprimer_feature(id_alti)
        if len(self.last_point) > 0:
            self.last_point.pop()
            self.last_z.pop()

    def display_resultats(self, infosResultats):
        QgsMessageLog.logMessage("", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Numéro de point : {}".format(infosResultats.id), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Coordonnées plani : {}".format(infosResultats.plani2D), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Altitude : {}".format(infosResultats.altitude), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Ecart plani : {}".format(infosResultats.ecart_plani), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Ecart alti : {}".format(infosResultats.ecart_alti), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Nombre d'images : {}".format(infosResultats.nb_images), tag="Infos Samon", level=Qgis.MessageLevel.Info)
        QgsMessageLog.logMessage("Résidus : {}".format(infosResultats.residu), tag="Infos Samon", level=Qgis.MessageLevel.Info)

    def display_point(self, point, mouse_button):
        tic = time.time()
        print_log("{}, {}".format(point.x(), point.y()))

        # On instancie un objet point avec les coordonnées du clic
        p = Point(point.x(), point.y())

        z_min = None
        z_max = None

        if len(self.last_point) >= 1 and self.mode == "linestring":
            distance = np.sqrt((p.x - self.last_point[-1].x)**2 + (p.y - self.last_point[-1].y)**2)
            delta_z = distance * np.tan(20*np.pi/180)
            
            z_min = self.last_z[-1] - delta_z
            z_max = self.last_z[-1] + delta_z
            QgsMessageLog.logMessage("delta_z : {}, {}, {}".format(delta_z, z_min, z_max), tag="Infos Samon", level=Qgis.MessageLevel.Info)

        elif len(self.last_point) >= 1 and self.mode == "polygon":
            z_min = self.last_z[-1] - 10
            z_max = self.last_z[-1] + 10
            QgsMessageLog.logMessage("delta_z : {}, {}".format(z_min, z_max), tag="Infos Samon", level=Qgis.MessageLevel.Info)

        

        if mouse_button == 1:
            self.projet.run(p, z_min=z_min, z_max=z_max, meme_bande=self.meme_bande)
            if self.projet.infosResultats.reussi:
                self.ajouter_point(self.projet.infosResultats)
                self.sauvegarder_shapefile()
                altitude = self.projet.infosResultats.altitude
                self.last_z.append(altitude)
                point2d = self.projet.infosResultats.point2d
                self.last_point.append(point2d)
                self.display_resultats(self.projet.infosResultats)
            else:
                QgsMessageLog.logMessage("Le calcul n'a pas abouti", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        else:
            self.geom_initialisation()
            self.last_z = []
            self.last_point = []
            QgsMessageLog.logMessage("", tag="Infos Samon", level=Qgis.MessageLevel.Info)
            QgsMessageLog.logMessage("Nouvelle géométrie", tag="Infos Samon", level=Qgis.MessageLevel.Info)


        self.iface.mapCanvas().refreshAllLayers()
        if self.nb_points >= 0:#On ne prend pas en compte le premier point qui est beaucoup plus long que les autres à calculer
            self.somme_temps += time.time() - tic
        self.nb_points += 1
        if self.nb_points >= 1:
            QgsMessageLog.logMessage("Temps de calcul : {}".format(time.time() - tic), tag="Infos Samon", level=Qgis.MessageLevel.Info)


    def key_point_pressed(self):
        QgsMessageLog.logMessage("Mode point", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        self.mode = "point"
    
    def key_polyligne_pressed(self):
        QgsMessageLog.logMessage("Mode linestring", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        self.mode = "linestring"
    
    def key_polygone_pressed(self):
        QgsMessageLog.logMessage("Mode polygon", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        self.mode = "polygon"
    
    def key_meme_bande_pressed(self):
        if self.meme_bande:
            QgsMessageLog.logMessage("Mode meme_bande arrêté", tag="Infos Samon", level=Qgis.MessageLevel.Info)
            self.meme_bande = False
        else:
            QgsMessageLog.logMessage("Mode meme_bande", tag="Infos Samon", level=Qgis.MessageLevel.Info)
            self.meme_bande = True
   
    def key_delete_pressed(self):
        QgsMessageLog.logMessage("Suppression du dernier point", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        self.geom_supprimer_dernier_point()
        print_log("On sauvegarde la géométrie")
        self.sauvegarder_shapefile()
    
    def key_saisie_pva(self):
        if self.projet.chantier is not None:
            QgsMessageLog.logMessage("Saisie sur orthos", tag="Infos Samon", level=Qgis.MessageLevel.Info)
            dialog = OrthoLocalesDialog(self.projet, self)
            dialog.exec()
            """if self.projet.infosResultats.reussi:
                self.geom_supprimer_dernier_point()
                self.ajouter_point(self.projet.infosResultats)
                self.sauvegarder_shapefile()
                altitude = self.projet.infosResultats.altitude
                self.last_z.append(altitude)
                point2d = self.projet.infosResultats.point2d
                self.last_point.append(point2d)
                self.display_resultats(self.projet.infosResultats)"""

    def key_choix_orthos(self):
        if self.projet.chantier is not None:
            dialog = ChoixOrthosDialog(self.projet)
            dialog.exec()
            for raster_layer in dialog.layers:
                QgsProject.instance().removeMapLayer(raster_layer)
            self.projet.lancer_calcul()
            if self.projet.infosResultats.reussi:
                self.geom_supprimer_dernier_point()
                self.ajouter_point(self.projet.infosResultats)
                self.sauvegarder_shapefile()
                altitude = self.projet.infosResultats.altitude
                self.last_z.append(altitude)
                point2d = self.projet.infosResultats.point2d
                self.last_point.append(point2d)
                self.display_resultats(self.projet.infosResultats)
            else:
                QgsMessageLog.logMessage("Le calcul n'a pas abouti", tag="Infos Samon", level=Qgis.MessageLevel.Info)


    def clic_sur_ortho_non_maitresse(self):
        if self.projet.infosResultats.reussi:
            self.geom_supprimer_dernier_point()
            self.ajouter_point(self.projet.infosResultats)
            self.sauvegarder_shapefile()
            altitude = self.projet.infosResultats.altitude
            self.last_z.append(altitude)
            point2d = self.projet.infosResultats.point2d
            self.last_point.append(point2d)
            self.display_resultats(self.projet.infosResultats)
        else:
            QgsMessageLog.logMessage("Le calcul n'a pas abouti", tag="Infos Samon", level=Qgis.MessageLevel.Info)

    
    def key_help(self):
        dialog = HelpDialog()
        dialog.exec()



    def run_prepa_chantier1(self):
        ta_path = self.dlg.lineEdit_prepa1_ta.text()
        emprise_path = visualisation_chantier(ta_path)
        iface.messageBar().pushMessage("Success !", "", level=Qgis.Success, duration=0)

        vlayer_DataLabeled = QgsVectorLayer(emprise_path, "emprise", "ogr")
        QgsProject.instance().addMapLayer(vlayer_DataLabeled, True)

        self.dlg.close()


    def run_prepa_chantier2(self):
        ta_path = self.dlg.lineEdit_prepa2_ta.text()
        selected_images_path = self.dlg.lineEdit_prepa2_selected_images.text()
        oriented_images_path = self.dlg.lineEdit_prepa2_oriented_images.text()
        storeref_path = self.dlg.lineEdit_prepa2_storeref.text()
        self.dlg.close()

        prepa_chantier_2(ta_path, selected_images_path, oriented_images_path, storeref_path)
        iface.messageBar().pushMessage("Success !", "", level=Qgis.Success, duration=0)

        


    def run_saisie(self):
        global pointTool
        chantier_path = self.dlg.lineEdit_saisie_chantier.text()
        type_correlation = str(self.dlg.comboBox.currentText())
        pva = os.path.join(chantier_path, "pvas")
        ortho = os.path.join(chantier_path, "ortho", "ortho.vrt")
        mnt = os.path.join(chantier_path, "mnt", "mnt.vrt")
        ta_name = [i for i in os.listdir(chantier_path) if i[-10:]=="adjust.XML"][0]
        ta = os.path.join(chantier_path, ta_name)
        resultats = os.path.join(chantier_path, "resultats")
        os.makedirs(resultats, exist_ok=True)
        QgsMessageLog.logMessage("Début", tag="Infos Samon", level=Qgis.MessageLevel.Info)

        ortho_layer = QgsRasterLayer(ortho, "ortho")
        QgsProject.instance().addMapLayer(ortho_layer)

        self.dlg.close()


        self.projet = Monoscopie(
                pva=pva, 
                ortho=ortho, 
                mnt=mnt, 
                ta_xml=ta, 
                resultats=resultats,
                type_correlation=type_correlation
            )

        QgsMessageLog.logMessage("Connection", tag="Infos Samon", level=Qgis.MessageLevel.Info)
        self.shortcut_point = QShortcut(Qt.Key.Key_P, self.iface.mainWindow())
        self.shortcut_point.setContext(Qt.ApplicationShortcut)
        self.shortcut_point.activated.connect(self.key_point_pressed)
            
        self.shortcut_polyligne = QShortcut(Qt.Key.Key_L, self.iface.mainWindow())
        self.shortcut_polyligne.setContext(Qt.ApplicationShortcut)
        self.shortcut_polyligne.activated.connect(self.key_polyligne_pressed)
            
        self.shortcut_polygone = QShortcut(Qt.Key.Key_G, self.iface.mainWindow())
        self.shortcut_polygone.setContext(Qt.ApplicationShortcut)
        self.shortcut_polygone.activated.connect(self.key_polygone_pressed)

        self.shortcut_delete= QShortcut(Qt.Key.Key_Z, self.iface.mainWindow())
        self.shortcut_delete.setContext(Qt.ApplicationShortcut)
        self.shortcut_delete.activated.connect(self.key_delete_pressed)

        self.shortcut_meme_bande = QShortcut(Qt.Key.Key_B, self.iface.mainWindow())
        self.shortcut_meme_bande.setContext(Qt.ApplicationShortcut)
        self.shortcut_meme_bande.activated.connect(self.key_meme_bande_pressed)

        self.shortcut_saisie_pva = QShortcut(Qt.Key.Key_S, self.iface.mainWindow())
        self.shortcut_saisie_pva.setContext(Qt.ApplicationShortcut)
        self.shortcut_saisie_pva.activated.connect(self.key_saisie_pva)

        self.shortcut_saisie_pva = QShortcut(Qt.Key.Key_D, self.iface.mainWindow())
        self.shortcut_saisie_pva.setContext(Qt.ApplicationShortcut)
        self.shortcut_saisie_pva.activated.connect(self.key_choix_orthos)

        self.shortcut_saisie_pva = QShortcut(Qt.Key.Key_H, self.iface.mainWindow())
        self.shortcut_saisie_pva.setContext(Qt.ApplicationShortcut)
        self.shortcut_saisie_pva.activated.connect(self.key_help)


        # On crée les trois couches qui contiendront les résultats
        self.geom_point = LayerGeometry("point", "point", QgsProject.instance().transformContext(), resultats)
        self.geom_linestring = LayerGeometry("linestring", "linestring", QgsProject.instance().transformContext(), resultats)
        self.geom_polygon = LayerGeometry("polygon", "polygon", QgsProject.instance().transformContext(), resultats)
        self.geom_altitude = GeometryLabelZ(QgsProject.instance().transformContext())
        
        

        # On ajoute dans l'interface la couche avec les coordonnées 2D du point cliqué et les coordonnées 2D du point après calcul
        QgsProject.instance().addMapLayer(self.geom_point.vl_2D)
        QgsProject.instance().addMapLayer(self.geom_point.vl_3D)
        QgsProject.instance().addMapLayer(self.geom_linestring.vl_2D)
        QgsProject.instance().addMapLayer(self.geom_linestring.vl_3D)
        QgsProject.instance().addMapLayer(self.geom_polygon.vl_2D)
        QgsProject.instance().addMapLayer(self.geom_polygon.vl_3D)
        QgsProject.instance().addMapLayer(self.geom_altitude.vl_3D)


        # A chaque clic sur le canvas, on lance le calcul dans la fonction self.display_point
        canvas = self.iface.mapCanvas()
        pointTool = QgsMapToolEmitPoint(canvas)
        pointTool.canvasClicked.connect(self.display_point)
        canvas.setMapTool(pointTool)




    def run(self):
        
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = SamonDialog()
            self.dlg.pushButton_prepa1_ta.clicked.connect(lambda  : self.select_file(self.dlg.lineEdit_prepa1_ta, "Select ta", "XML files (*.xml *.XML)"))
            self.dlg.pushButton_prepa2_ta.clicked.connect(lambda  : self.select_file(self.dlg.lineEdit_prepa2_ta, "Select ta", "XML files (*.xml *.XML)"))
            self.dlg.pushButton_prepa2_selected_images.clicked.connect(lambda  : self.select_file(self.dlg.lineEdit_prepa2_selected_images, "Select selected images", "Shapefile (*.shp)"))
            self.dlg.pushButton_prepa2_oriented_images.clicked.connect(lambda  : self.select_directory(self.dlg.lineEdit_prepa2_oriented_images, "Select oriented images"))
            self.dlg.pushButton_prepa2_storeref.clicked.connect(lambda  : self.select_directory(self.dlg.lineEdit_prepa2_storeref, "Select storeref"))
            self.dlg.pushButton_saisie_chantier.clicked.connect(lambda  : self.select_directory(self.dlg.lineEdit_saisie_chantier, "Select chantier"))

            self.dlg.pushButton_prepa1_run.clicked.connect(self.run_prepa_chantier1)
            self.dlg.pushButton_prepa2_run.clicked.connect(self.run_prepa_chantier2)
            self.dlg.pushButton_saisie_run.clicked.connect(self.run_saisie)

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        if result:
            pass
