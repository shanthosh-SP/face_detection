import face_recognition
import cv2 as cv
import numpy as np

imgDhoni=face_recognition.load_image_file("photos/148841-clraytzonv-1602133322.jpg")
imgDhoni=cv.cvtColor(imgDhoni,cv.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("photos/m7opt04g_ms-dhoni-afp_625x300_06_July_20.jpg")
imgTest=cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgDhoni)[0]
encodeDhoni=face_recognition.face_encodings(imgDhoni)[0]
cv.rectangle(imgDhoni,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),3)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


results=face_recognition.compare_faces([encodeDhoni],encodeTest)
facedis=face_recognition.face_distance([encodeDhoni],encodeTest)


print(results)
print(facedis)

cv.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

cv.imshow("Thala",imgDhoni)
cv.imshow("thala",imgTest)
cv.waitKey(0)

