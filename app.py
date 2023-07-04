from flask import Flask, render_template,request,Response;
from combined import camera_stream;

app = Flask(__name__)

def generate_frames():
    while True:
            
        ## read the camera frame
        frame = camera_stream()
        if frame is not None:
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')



@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/feed/', methods=['GET'])
def feed():
    return render_template('main2.html')



if __name__ == '__main__':
    app.run(port=3000,debug=True)