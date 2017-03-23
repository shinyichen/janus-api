import urllib2
import requests

print("initialize...")
print(urllib2.urlopen("http://localhost:8080/initialize").read())

print("create template...")
# payload = [{'image_path':'/nfs/div2/jchen/janus-dev/janus-api/src/cserver/test_imgs/example1.png', 'face_x': 39, 'face_y': 24, 'face_width': 175, 'face_height': 204},
#     {'image_path':'/nfs/div2/jchen/janus-dev/janus-api/src/cserver/test_imgs/example2.png', 'face_x': 360, 'face_y': 45, 'face_width': 726, 'face_height': 810},
#     {'image_path':'/nfs/div2/jchen/janus-dev/janus-api/src/cserver/test_imgs/example3.png', 'face_x': 248, 'face_y': 43, 'face_width': 330, 'face_height': 360}]
payload = [{'image_path':"/lfs2/glaive/data/CS3_2.0/frames/10003.png", 'face_x': 638, 'face_y': 82, 'face_width': 185, 'face_height': 212},
    {'image_path':"/lfs2/glaive/data/CS3_2.0/frames/10004.png", 'face_x': 627, 'face_y': 93, 'face_width': 170, 'face_height': 206},
    {'image_path':"/lfs2/glaive/data/CS3_2.0/frames/10005.png", 'face_x': 597, 'face_y': 134, 'face_width': 175, 'face_height': 201}]
r = requests.post("http://localhost:8080/search", json=payload)
print(r.status_code)
print(r.text)

#result = r.json()
# print(result[0]['template_id'])

print("finalize...")
print(urllib2.urlopen("http://localhost:8080/finalize").read())
