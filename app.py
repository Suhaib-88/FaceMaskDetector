from flask import Flask, render_template, Response,request
import cv2
from camera import Video
import warnings

warnings.filterwarnings('ignore')

app=Flask(__name__)

Cam=cv2.VideoCapture(0)
    
def gen_vids(camera):  
    while True:
        frame=camera.get_vids(cam=Cam)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_vids(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)
