
from pysocle.photogrammetry.shot import Shot
import numpy as np
from osgeo import gdal
import os
from scipy import ndimage


class EpipolarGeometry:

    def __init__(self, image1, image2, dem, pva) -> None:
        """
        Construit deux images dans leur géométrie épipolaire

        Equations issues de : Mathématiques de la photogrammétrie numérique, J. F. Haas, 2004
        """

        self.image1:Shot = image1
        self.image2:Shot = image2
        self.dem = dem
        self.pva = pva

        self.dh = 0

        self.r2e:np.array = None
        self.r1e:np.array = None

        # On détermine la matrice permettant de passer l'image 1 en géométrie épipolaire
        self.r1e = self.geom_epipolaire(self.image1, self.image2)
        # On détermine la matrice permettant de passer l'image 2 en géométrie épipolaire
        self.r2e = self.geom_epipolaire(self.image2, self.image1, facteur_base=-1)

        # Contrôle pour vérifier que les fonctions image_to_epip et epip_to_image permettent de retrouver le point initial
        """c, l = self.image_to_epip(np.array([100]), np.array([100]), self.image1, self.r1e, use_dh=True)
        c_b, l_b = self.epip_to_image(c, l, self.image1, self.r1e, use_dh=True)"""


    def image_to_epip(self, c, l, image:Shot, E, use_dh=True):
        """
        Convertit les coordonnées images d'une image en coordonnées épipolaires
        """

        focale = -image.imc.camera.focal

        c_shot = c - self.image1.imc.camera.x_ppa
        l_shot = l - self.image1.imc.camera.y_ppa
        
        
        m = np.vstack([c_shot, l_shot, np.zeros(c_shot.shape)])
        F = np.full_like(m, np.array([[0], [0], [focale]]))
        L1E = E[0,:].T
        L2E = E[1,:].T
        L3E = E[2,:].T

        F_prime = E @ F
        p_prime = F_prime[2]
        x_c_prime = F_prime[0]
        y_c_prime = F_prime[1]

        m_f = m-F
        x = x_c_prime - p_prime * ((L1E @ (m_f)) / (L3E @ (m_f)))
        if use_dh:
            y = y_c_prime + self.dh-p_prime * ((L2E @ (m_f)) / (L3E @ (m_f)))
        else:
            y = y_c_prime - p_prime * ((L2E @ (m_f)) / (L3E @ (m_f)))

        return x, y



    def epip_to_image(self, c, l, image:Shot, E, use_dh=True):
        """
        Convertit les coordonnées épipolaires d'une image en coordonnées images
        """

        focale = -image.imc.camera.focal

        if use_dh:
            l -= self.dh

        # On met en forme les points
        m = np.vstack([c, l, np.zeros(c.shape)])

        # On calcule F_prime dans le repère de l'image épipolaire
        # F_prime est le point F mais dans le repère de l'image épipolaire
        F = np.full_like(m, np.array([[0], [0], [focale]]))
        F_prime = E @ F


        C1E = E[:,0]
        C2E = E[:,1]
        C3E = E[:,2]

        m_f = m-F_prime
        x = -focale * ((C1E @ (m_f)) / (C3E @ (m_f)))
        y = -focale * ((C2E @ (m_f)) / (C3E @ (m_f)))

        return x + image.imc.camera.x_ppa, y + image.imc.camera.y_ppa




    def geom_epipolaire(self, im1, im2, facteur_base=1):
        """
        Calcule E, la matrice pour passer en géométrie épipolaire
        """
        # Calcul de omega

        R = im1.imc.mat_eucli
        RA = im2.imc.mat_eucli

        L2 = R[1,:]
        L3A = RA[2,:]
        L1 = R[0,:]

        t = (L2 @ L3A.T) / (L1 @ L3A.T)

        a = - t / (np.sqrt(1+t**2))
        b = 1 / np.sqrt(1+t**2)
        c = 0


        # Calcul de n
        L3 = R[2,:]
        s1 = im1.imc.sommet
        s2 = im2.imc.sommet
        B = np.array([s2[0] - s1[0], s2[1] - s1[1], s2[2] - s1[2]]) * facteur_base
        d = np.sqrt((1+t**2) * (L3 @ B)**2 + (L1 @ B + t * L2 @ B)**2)
        n = np.array([L3 @ B / d, t * L3 @ B / d, -(L1 @ B + t * L2 @ B) / d])
        
        # Evite une inversion des images entre épipolaire et non épipolaire
        if n[2] < 0:
            n = -n

        # Calcul de theta
        cos_theta = n[2]
        sin_theta = np.sqrt(n[0]**2 + n[1]**2) * (L3 @ B / np.abs(L3 @ B))

        # Calcul de R_prime
        omega_axiateur = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
        R_prime = np.eye(3) + omega_axiateur * sin_theta + omega_axiateur @ omega_axiateur * (1 - cos_theta)

        # Calcul de R_seconde
        L1_prime = R_prime[0,:]
        L2_prime = R_prime[1,:]
        sigma = 1
        B_norme = np.linalg.norm(B)
        R_seconde = 1 / B_norme * np.array([[sigma * L1_prime @ R @ B, sigma * L2_prime @ R @ B, 0], [-sigma * L2_prime @ R @ B, sigma * L1_prime @ R @ B, 0], [0, 0, B_norme]])
        # calcul de E, matrice pour passer dans la géométrie épipolaire
        E = R_seconde @ R_prime
        return E


    def calcul_dh(self, centre):
        """
        Calcule dh, la différence en y pour que dans la géométrie épipolaire, les lignes des deux images correspondent parfaitement

        On calcule dh uniquement autour du centre car il semblerait que ce h varie légèrement selon où l'on se trouve sur la photo
        """
        c = np.linspace(centre[0]-50, centre[1]+50, 5)
        l = np.linspace(centre[0]-50, centre[1]+50, 5)
        c_grid, l_grid = np.meshgrid(c, l)
        c_grid = c_grid.reshape((-1))
        l_grid = l_grid.reshape((-1))

        x_world, y_world, z_world = self.image1.imc.image_to_world(c_grid, l_grid, self.dem)

        c_im2, l_im2 = self.image2.imc.world_to_image(x_world, y_world, z_world)

        x1, y1 = self.image_to_epip(c_grid, l_grid, self.image1, self.r1e, use_dh=False)
        x2, y2 = self.image_to_epip(c_im2, l_im2, self.image2, self.r2e)
        
        return np.mean(y1 - y2)