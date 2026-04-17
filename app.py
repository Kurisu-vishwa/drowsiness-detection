import cv2
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained = False)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 4)
)
model.load_state_dict(torch.load("drowsiness_model.pth", map_location=device))

model.to(device)
model.eval()



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                     + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                    + "haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

class_names = ['Closed', 'no_yawn', 'Open', 'yawn']
drowsy_counter = 0

while True:
    ret, frame =  cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for(x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        
        for(ex, ey, ew, eh) in eyes:
            input_img = face
            input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            inputt = transform(input_rgb).unsqueeze(0).to(device)
            
        
            with torch.no_grad():
                outputs = model(inputt)
                _, pred = torch.max(outputs, 1)
                label = class_names[pred.item()]
                if label in ["Closed", "yawn"]:
                    drowsy_counter += 1
                else:
                    drowsy_counter = 0
                cv2.putText(frame, f"counter:{drowsy_counter}", (20, 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                if drowsy_counter > 20:
                    cv2.putText(frame, "Drowsy Alert !!!", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)

            
        cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

            
        cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        break  

    cv2.imshow("Eye Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
