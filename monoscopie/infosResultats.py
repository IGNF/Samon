from shapely.geometry import Point
import numpy as np

class InfosResultats():

    def __init__(self, reussi:bool, id:int=None, plani2D:Point=None, mnt:float=None, point3d:np.array=None, nb_images:int=None, residu:float=None) -> None:
        self.reussi = reussi
        if self.reussi:
            self.id:int = id
            self.point2d:Point = plani2D
            self.point3d:np.array = point3d
            self.plani2D:str = "({}, {})".format(point3d[0], point3d[1])
            self.altitude:float = point3d[2]
            self.ecart_plani:float = np.sqrt((plani2D.x-point3d[0])**2 + (plani2D.y-point3d[1])**2)
            self.ecart_alti:float = point3d[2] - mnt
            self.nb_images:int = nb_images
            self.residu:float = residu
        self.ecart_z_lidar:float = None

    def set_ecart_z_lidar(self, z_lidar):
        self.ecart_z_lidar = self.altitude - z_lidar
