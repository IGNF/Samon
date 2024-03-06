from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel

class HelpDialog(QDialog):

    def __init__(self) -> None:
        QDialog.__init__(self)

        layout = QVBoxLayout()

        h = QLabel("h : aide")
        layout.addWidget(h)

        p = QLabel("p : mode point")
        layout.addWidget(p)

        l = QLabel("l : mode ligne")
        layout.addWidget(l)

        g = QLabel("g : mode polygone")
        layout.addWidget(g)

        clic_droit = QLabel("clic droit : fermer une figure et en commencer une nouvelle. ne sert que pour les modes ligne ou polygone")
        layout.addWidget(clic_droit)

        z = QLabel("z : supprimer le dernier point. On ne peut supprimer que les points de la figure en cours")
        layout.addWidget(z)

        b = QLabel("b : on utilise uniquement les pvas de la même bande que la va maitresse. Utile pour les bâtiments")
        layout.addWidget(b)

        s = QLabel("s : possibilité de saisir un point sur une des orthos locales du point dernièrement saisi qui sera alors remplacé")
        layout.addWidget(s)

        d = QLabel("d : possibilité de choisir les pvas utilisées pour le calcul. Dès que l'état d'une checkbox est modifié, le calcul est relancé")
        layout.addWidget(d)

        self.setLayout(layout)