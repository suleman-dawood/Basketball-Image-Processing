import cv2
import os
from vidgear.gears import CamGear

train_URLs = [
    "https://www.youtube.com/watch?v=Y0BM6y25N9s", 
    "https://www.youtube.com/watch?v=tBn8E4LQGMM",
    "https://www.youtube.com/watch?v=SYon2G1gFgE&t=13s"]

numerator = 0
output_dir = "D:/StudyRelated/Machine Learning Projects/NBA/dataset/images"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#loop through each URL in the list
for url in train_URLs:
    print(f"Processing video: {url}")
    stream = CamGear(source=url, stream_mode=True, logging=True).start()
    
    #process video frames
    while True:
        frame = stream.read()
        numerator += 1
        print(f"Extracting frame {numerator} from video.")
        if frame is None:
            break
        
        image_path = os.path.join(output_dir, f"frame_{numerator:05d}.png")
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized)

        #display frame (optional)
        cv2.putText(frame, f"Frame no: {numerator}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        cv2.imshow("Frame Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.stop()