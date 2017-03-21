import urllib2
import requests

print("initialize...")
print(urllib2.urlopen("http://localhost:18080/initialize").read())

print("create template...")
payload = [{'image_path':'example1.jpeg', 'face_x': 1, 'face_y': 2, 'face_width': 3, 'face_height': 4},
    {'image_path':'example2.jpeg', 'face_x': 5, 'face_y': 6, 'face_width': 7, 'face_height': 8},
    {'image_path':'example3.jpeg', 'face_x': 9, 'face_y': 10, 'face_width': 11, 'face_height': 12}]
r = requests.post("http://localhost:18080/search", json=payload)
print(r.status_code)
print(r.text)

result = r.json()
print(result[0]['template_id'])

print("finalize...")
print(urllib2.urlopen("http://localhost:18080/finalize").read())
