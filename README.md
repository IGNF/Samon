# Samon

Plugin Qgis pour la saisie monoscopique. L'objectif est de calculer les coordonnées 3D du point cliqué sur la BD Ortho en corrigeant notamment le dévers. 



# Installation

* QGIS/Extensions/Installer les extensions
* Sélectionner Samon

Dans le cas où il y a des problèmes de librairies qui ne sont pas installées :
## Windows
* Ouvrir OSGeo4W Shell (dans le menu démarrer de Windows)
* python -m pip install [librairie]

## Linux
* QGIS/Extension/Console Python
* import pip
* pip.main(["install", "librairie"])