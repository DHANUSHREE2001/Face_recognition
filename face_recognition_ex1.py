import cv2
import face_recognition
import numpy as np



# 2 images compare
imgelon = face_recognition.load_image_file('D:\BTech_CSE_2019-2023\projects\SIH\images\elonmusk.jpg')
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
encodeElon = face_recognition.face_encodings(imgelon)[0]

imgjane = face_recognition.load_image_file('D:\BTech_CSE_2019-2023\projects\SIH\images\candice.jpg')
imgjane = cv2.cvtColor(imgjane,cv2.COLOR_BGR2RGB)
encodejane = face_recognition.face_encodings(imgjane)[0]

imgme = face_recognition.load_image_file('D:\BTech_CSE_2019-2023\projects\SIH\images\me.jpeg')
imgme = cv2.cvtColor(imgme,cv2.COLOR_BGR2RGB)
encodeme = face_recognition.face_encodings(imgme)[0]

imgme1 = face_recognition.load_image_file('D:\BTech_CSE_2019-2023\projects\SIH\images\dd.jpg')
imgme1 = cv2.cvtColor(imgme,cv2.COLOR_BGR2RGB)
encodeme1 = face_recognition.face_encodings(imgme1)[0]

#faceLoc = face_recognition.face_locations(imgelon)[0] #returns 4 vals
#face_landmarks_list = face_recognition.face_landmarks(imgelon) #facial features

#print(faceLoc)
#print(face_landmarks_list)

#cv2.rectangle(imgelon,() )

#results = face_recognition.compare_faces([encodeElon], encodejane)

#if results[0] == True:
    #print("Match!")
#else:
    #print("Mis-Match!")

#cv2.imshow('Elon Musk',imgelon)
#cv2.imshow('Jane Doe',imgjane)
#cv2.waitKey(0)

###########

known_face_encodings = [
    encodeElon,
    encodejane,
    encodeme,
    encodeme1
]

known_face_names = [
    "Elon musk",
    "Candice",
    "Dhanu",
    "Girl"
]
process_this_frame = True
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame  =video_capture.read()
    #reduce frame size for faster recognition processing
    small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    rgb_small_frame = small_frame[:,:,::-1] #height*width*3 - rgb
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

