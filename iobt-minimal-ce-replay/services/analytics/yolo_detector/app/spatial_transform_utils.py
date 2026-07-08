import torch
import pandas as pd
from scipy.spatial.transform import Rotation
import torch.nn as nn
import copy

def meters_per_pixel(lat, zoom):
    #reference: https://medium.com/techtrument/how-many-miles-are-in-a-pixel-a0baf4611fff
    #Reference does not seem to account for scale=2 when getting maps
    #Needs zoom+1, no zomm
    return (156543.03392 * torch.cos(torch.tensor(lat * torch.pi / 180.0)) / (torch.pow(torch.tensor(2.0), 1+zoom)))

def euler_to_rot(roll, pitch, yaw):
    rot = Rotation.from_euler('xzy', [pitch.item(), yaw.item(), roll.item()])
    return torch.tensor(rot.as_matrix()).float()

def to_implicit(lat, lon, zoom):
    implicit_height = (2*256)*(2**zoom)
    implicit_width  = (2*256)*(2**zoom)
    
    R = implicit_width/(2*torch.pi)
    FE = 180.0
    lonRad = torch.deg2rad(lon + FE)
    implicit_x = lonRad * R
    
    latRad = torch.deg2rad(lat)
    verticalOffsetFromEquator = R * torch.log(torch.tan(torch.pi / 4 + latRad / 2))
    implicit_y = implicit_height / 2 - verticalOffsetFromEquator
    
    return implicit_x, implicit_y

def to_pixel(lat, lon, center_lat, center_lon, zoom, width, height):
    cix, ciy = to_implicit(center_lat, center_lon, zoom)
    ix, iy = to_implicit(lat, lon, zoom)
    x = (ix - cix) + width/2.0
    y = (iy - ciy) + height/2.0
    return x, y

def gps_to_meters(points,center_lat=39.351,center_lon=-76.345,zoom=18,map_width=800,map_height=800,type=""):
    mppx   = meters_per_pixel(center_lat, zoom)
    x, y   = to_pixel(points[:,[0]], points[:,[1]], torch.tensor(center_lat), torch.tensor(center_lon), zoom=zoom, width=map_width, height=map_height)
    output = torch.hstack([(x-map_width/2)*mppx,-1*(y-map_height/2)*mppx,points[:,[2]]])
    return output.float()

def parse_gps_log(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = [eval(l.strip()) for l in lines]
    df   = pd.DataFrame(data)
    pos  = torch.tensor([[df['lt'].mean(), df['ln'].mean(), df['al'].mean()]],dtype=torch.float64)
    std  = torch.tensor([[df['lt'].std(), df['ln'].std(), df['al'].std()]])
    return pos, std

class point_projector(nn.Module):
    def __init__(self,node_info):
        super().__init__()

        self.info = copy.copy(node_info)

        self.X = nn.Parameter(torch.tensor(node_info["location"]["X"]).float(),requires_grad=True)
        self.Y = nn.Parameter(torch.tensor(node_info["location"]["Y"]).float(),requires_grad=True)
        self.Z = nn.Parameter(torch.tensor(node_info["location"]["Z"]).float(),requires_grad=True)

        self.dX = nn.Parameter(torch.tensor(node_info["zed"]["location_offset_X"]).float())
        self.dY = nn.Parameter(torch.tensor(node_info["zed"]["location_offset_Y"]).float())
        self.dZ = nn.Parameter(torch.tensor(node_info["zed"]["location_offset_Z"]).float())

        self.pitch = nn.Parameter(torch.tensor(node_info["zed"]["pitch"]).float())
        self.yaw = nn.Parameter(torch.tensor(node_info["zed"]["yaw"]).float())
        self.roll = nn.Parameter(torch.tensor(node_info["zed"]["roll"]).float())

        self.set_camera()

    def get_info(self):
        self.info["zed"]["yaw"] = self.yaw.detach().item()
        self.info["zed"]["pitch"] = self.pitch.detach().item()
        self.info["zed"]["roll"] = self.roll.detach().item()
        self.info["location"]["X"] = self.X.detach().item()
        self.info["location"]["Y"] = self.Y.detach().item()
        self.info["location"]["Z"] = self.Z.detach().item() 
        return self.info
    
    def set_camera(self):

        r = euler_to_rot(self.roll, self.pitch,self.yaw).float()
        t = torch.hstack([self.X,self.Y,self.Z]).reshape(3,1).float()
        d = torch.hstack([self.dX,self.dY,self.dZ]).reshape(3,1).float()

        R = r.T
        T = -R@t - d

        self.R=R
        self.T=T

    def forward(self,points_world):
        self.set_camera()
        return self.world_to_image(points_world)

    def world_to_local(self,points_world):
        return (self.R@points_world.T + self.T).T

    def local_to_image(self,points_local):
        points_img = torch.zeros(points_local.shape)
        points_img[:,0] = self.info["zed"]["cx"] + self.info["zed"]["fx"]*points_local[:,0]/points_local[:,1]
        points_img[:,1] = self.info["zed"]["cy"] - self.info["zed"]["fy"]*points_local[:,2]/points_local[:,1]
        points_img[:,2] = points_local[:,1]
        return(points_img)

    def world_to_image(self,points_world):
        points_local = self.world_to_local(points_world)
        return(self.local_to_image(points_local))

    def image_to_local(self,points_img):
        points_local = torch.zeros(points_img.shape)
        points_local[:,1] = points_img[:,2]
        points_local[:,0] = (points_img[:,0] - self.info["zed"]["cx"])/self.info["zed"]["fx"]*points_local[:,1]
        points_local[:,2] = -1*(points_img[:,1] - self.info["zed"]["cy"])/self.info["zed"]["fy"]*points_local[:,1]
        return(points_local)

    def local_to_world(self, points_local):
        return (self.R.T@(points_local.T-self.T)).T

    def image_to_world(self,points_img):
        points_local = self.image_to_local(points_img)
        points_world = self.local_to_world(points_local)
        return(points_world)

    def local_to_world_cov(self,cov):
        return self.R @ cov @ self.R.T   
     


