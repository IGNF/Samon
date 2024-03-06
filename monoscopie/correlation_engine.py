from .orthoLocale import OrthoLocale
import numpy as np
from typing import Tuple, Union
from .pva import Pva

class CorrelationEngine:

    def __init__(self, ortholocale:Union[OrthoLocale, Pva], size_window:int, reference:np.array, v2=True) -> None:
        """
        Deux manières de faire de la corrélation :
        - v2 = False : Pour chaque coordonnées x, y, on fait un rééchantillonage bilinéaire de orthoLocale et on fait la corrélation
        - v2 = True : on récupère l'ortholocale sans aucun rééchantillonnage, puis autour du point de corrélation maximum, 
        on établit une courbe ax**2+bx+c pour laquelle on récupère le maximum. Ne fonctionne qu'avec la corrélation sur pva sans géométrie épipolaire
        """
        
        self.orthoLocale = ortholocale
        self.size_window = size_window
        self.reference = reference
        self.v2 = v2

        self.resolution = 0.05
        self.size_window_accurate = 6
    

    def run(self, x:np.array, y:np.array) -> Tuple[float, float, float]:
        """
        Cherche la corrélation autour des points x, y

        Renvoie les coordonnées (x_chap, y_chap) du meilleur point de corrélation, avec le score de corrélation
        """

        x_chap, y_chap, correlation = 0, 0, 0

        # Effectue une première corrélation
        if self.v2 : 
            x_chap, y_chap, correlation = self.run_correlation_v2(x, y)
        else:
            x_chap, y_chap, correlation = self.run_correlation(x, y)

        # Si le score de corrélation est inférieur à 0.9, alors on fait la corrélation sur l'ortho retournée
        if correlation < 0.9:
            if self.v2:
                x_chap_2, y_chap_2, correlation_2 = self.run_correlation_v2(x, y, flip=True)
            else:
                x_chap, y_chap, correlation_2 = self.run_correlation(x, y)

            # On renvoie les coordonnées associées à la meilleure corrélation
            if correlation_2 > correlation:
                return x_chap_2, y_chap_2, correlation_2

        # Parfois la corrélation est supérieure à 1 (sans doute parce qu'en v2, le score de corrélation est meilleur sur un
        #  des pixels à côté de celui appartenant à la droite épipolaire et avec le meilleur score de corrélation)
        if correlation > 1 or np.isnan(correlation):
            correlation = 0
        
        return x_chap, y_chap, correlation



    def run_correlation_v2(self, x, y, flip=False) -> Tuple[float, float, float]:

        # On convertit x, y en int et on supprime les doublons
        x_unique, y_unique = self.get_unique(x, y)

        # On récupère le score de corrélation pour tous les (x_unique, y_unique)
        correlation = self.compute_correlation_without_argmax(x_unique, y_unique, flip)

        # On récupère le score de corrélation maximum et les coordonnées du point qui corrèle le mieux
        indice = np.argmax(correlation)
        correlation_max = np.max(correlation)
        x_chap = x_unique[indice]
        y_chap = y_unique[indice]

        # On va calculer le score de corrélation pour les quatre pixels à côté du point qui corrèle le mieux  
        # On spécifie les coordonnées des points
        x_around = np.array([[x_chap-1], [x_chap+1], [x_chap], [x_chap]])
        y_around = np.array([[y_chap], [y_chap], [y_chap-1], [y_chap+1]])

        # On récupère le score de corrélation pour les quatre pixels à côté du point qui corrèle le mieux  
        correlation_2 = self.compute_correlation_without_argmax(x_around, y_around, flip)
        
        # On résout le système d'équation pour obtenir le score de corrélation maximal ainsi que les coordonnées, cette fois en subpixellaire
        x_max, correlation1_max = self.solve_system(x_chap-1, x_chap, x_chap+1, correlation_2[0], correlation_max, correlation_2[1])
        y_max, correlation2_max = self.solve_system(y_chap-1, y_chap, y_chap+1, correlation_2[2], correlation_max, correlation_2[3])

        return x_max, y_max, (correlation1_max+correlation2_max)/2
    
    
    def compute_correlation_without_argmax(self, x:np.array, y:np.array, flip:bool):

        # On crée pour chaque coordonnées (x, y) une vignette sans rééchantillonage
        orthos = self.orthoLocale.create_vignette(x, y, self.size_window, 1)

        # Normalisation des orthos
        orthos_tild = (orthos - np.mean(orthos, axis=(1,2)).reshape((-1, 1, 1))) / np.std(orthos, axis=(1,2)).reshape((-1, 1, 1))
        
        # Si le retournement est demandé
        if flip:
            orthos_tild = np.flip(orthos_tild, axis=(1,2))
        
        # Normalisation de la référence
        bd_ortho = np.tile(self.reference, (orthos.shape[0], 1, 1))
        bd_ortho_tild = (bd_ortho - np.mean(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))) / np.std(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))
        
        # Calcul de la corrélation
        correlation = 1 - ( np.sum((bd_ortho_tild-orthos_tild)**2, axis=(1,2)) / orthos_tild.shape[1]**2)/2
        return correlation
    
    
    def get_unique(self, x:np.array, y:np.array):
        """
        Convertit x et y en entiers, et supprime les doublons
        """
        
        # Conversion en entiers
        x = (np.rint(x)).astype(int).reshape((-1, 1))
        y = (np.rint(y)).astype(int).reshape((-1, 1))

        x_m_1 = x - 1
        x_p_1 = x + 1
        y_m_1 = y - 1
        y_p_1 = y + 1

        x = np.concatenate((x_m_1, x_m_1, x_m_1, x, x, x, x_p_1, x_p_1, x_p_1), axis=0)
        y = np.concatenate((y_m_1, y, y_p_1, y_m_1, y, y_p_1, y_m_1, y, y_p_1), axis=0)

        # Suppression des doublons
        coordinates = np.concatenate((x, y), axis=1)
        coordinates_unique = np.unique(coordinates, axis=0)
        return coordinates_unique[:,0], coordinates_unique[:,1]
    
    
    
    def solve_system(self, x0, x1, x2, c0, c1, c2):
        """
        Résolution du système ax*2 + bx + c = corr avec :
            - x : coordonnées du point suivant un axe
            - corr : score de corrélation pour le point x
            - a, b et c : inconnues : paramètres de la fonction
        x et corr sont connus pour trois points : c0 = f(x0), c1 = f(x1), c2 = f(x2)
        """
        
        # Construction des matrices
        A = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]])
        B = np.array([[c0], [c1], [c2]])
        
        # résolution du système
        X = np.linalg.solve(A, B)
        a = X[0,0]
        b = X[1,0]
        c = X[2,0]

        # On récupère la coordonnée x pour laquelle la fonction f est maximale
        x_max = -b / (2*a)
        # On récupère la valeur maximale de la fonction
        correlation_max = -b**2/(4*a) + c
        return x_max, correlation_max


    def run_correlation(self, x:np.array, y:np.array, flip=False) -> Tuple[float, float, float]:
        #D'abord on cherche la corrélation sur un sous-échantillonnage de facteur 4
        x_sous_ech = x[::4]
        y_sous_ech = y[::4]

        #On obtient un premier centre de corrélation
        x_chap, y_chap, _ = self.compute_correlation(x_sous_ech, y_sous_ech, flip)

        #On construit des nouveaux tableaux x_array, y_array
        x_array, y_array = self.get_x_y_improved(x_chap, y_chap)

        #On fait la corrélation sur ces nouvelles valeurs de x, y
        x_chap, y_chap, correlation = self.compute_correlation(x_array, y_array, flip)

        #On renvoie le centre de corrélation final
        return x_chap, y_chap, correlation


    def compute_correlation(self, x:np.array, y:np.array, flip:bool)-> Tuple[float, float, float]:
        """
        Renvoie parmi les points (x, y) celui qui a le meilleur score de corrélation

        flip : dans le cas de la corrélation sur pvas, les pvas ne sont pas toutes orientées dans le même sens. 
        Si flip = True, alors les vignettes sont retournées afin d'être dans le même sens que l'image maîtresse
        """
        orthos = self.orthoLocale.create_small_ortho_numpy(x, y, self.size_window, 1)
        orthos = np.squeeze(orthos)
        orthos_tild = (orthos - np.mean(orthos, axis=(1,2)).reshape((-1, 1, 1))) / np.std(orthos, axis=(1,2)).reshape((-1, 1, 1))
        
        # Si le retournement est demandé
        if flip:
            orthos_tild = np.flip(orthos_tild, axis=(1,2))
        
        bd_ortho = np.tile(self.reference, (orthos.shape[0], 1, 1))
        bd_ortho_tild = (bd_ortho - np.mean(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))) / np.std(bd_ortho, axis=(1,2)).reshape((-1, 1, 1))
        
        correlation = 1 - ( np.sum((bd_ortho_tild-orthos_tild)**2, axis=(1,2)) / orthos_tild.shape[1]**2)/2

        indice = np.argmax(correlation)
        correlation_max = np.max(correlation)
        
        x_chap = x[indice]
        y_chap = y[indice]

        return x_chap, y_chap, correlation_max


    def get_x_y_improved(self, x:float, y:float) -> Tuple[np.array, np.array]:
        """
        Crée une zone de recherche autour de (x, y)
        """

        x_min = x - self.resolution * self.size_window_accurate
        x_max = x + self.resolution * self.size_window_accurate
        x_interval = np.linspace(x_min, x_max, 2*self.size_window_accurate+1)

        y_min = y - self.resolution * self.size_window_accurate
        y_max = y + self.resolution * self.size_window_accurate

        y_interval = np.flip(np.linspace(y_min, y_max, 2*self.size_window_accurate+1), axis=0)
        xv_temp, yv_temp = np.meshgrid(x_interval, y_interval)
        xv = xv_temp.reshape((-1))
        yv = yv_temp.reshape((-1))
        return xv, yv

