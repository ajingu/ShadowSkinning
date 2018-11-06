from pythonosc import udp_client
from time import sleep


class OSCclient:
    def __init__(self, ip, port, sleep_time=0.1):
        self.ip = ip
        self.port = port
        self.sleep_time = sleep_time
        self.client = udp_client.SimpleUDPClient(ip, port)
        
    def send(self, shadow, frame_index):
        joints = [[x[0], x[1]] for x in shadow.body_part_positions]
        triangles = shadow.triangle_vertex_indices
        vertices = [[int(x[0]), int(x[1])] for x in shadow.vertex_positions]
        
        self.client.send_message("/joints", joints)
        sleep(self.sleep_time)
        self.client.send_message("/triangles", triangles)
        sleep(self.sleep_time)
        self.client.send_message("/vertices", vertices)
        sleep(self.sleep_time)
        self.client.send_message("/frame_index", int(frame_index))