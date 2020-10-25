
import argparse
import json
from flask import Flask,request, redirect, url_for
from flask import render_template
import os
from collections import Counter
from werkzeug.utils import secure_filename
import tempfile
import requests
from worker import recv_Img
import pickle
import cv2
from threading import Lock, Thread


class FlaskServer():
    def __init__(self, port, max_persons ):
        self.app = Flask(__name__)
        self.max_persons=int(max_persons)
        self.port=port
        self.lock = Lock()
        self.UPLOAD_FOLDER = './static'
        self.app.config['UPLOAD_FOLDER'] = self.UPLOAD_FOLDER
        self.app.route('/', methods=['POST'])(self.show_img)
        self.app.route('/result',methods=['POST'])(self.get_imageinfo)
        self.video_map={}
        self.process_counter=0
        self.app.run(host='127.0.0.1' ,port=self.port, threaded=True)

    def show_img(self):
        if request.files['video']:
            video = request.files['video'].read()
            fp = tempfile.NamedTemporaryFile(dir=self.UPLOAD_FOLDER) #save file in directory uploads
            fp.write(video)
            fp.seek(0)
            vidcap = cv2.VideoCapture(fp.name)
            #count number of frames

            total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            success,image = vidcap.read()

            # check concorrency here
            self.lock.acquire()
            self.process_counter += 1
            proc = self.process_counter
            self.lock.release()

            #save for each video,the total of frames of it, and a counter of frames that were already processed by the worker(s)
            if proc not in self.video_map:
                self.video_map[proc] = {"total" : total, "count" : 0 , "classes": {}, "timestamp" : 0}

            count = 0 
            while success:
                data = {'proc': proc,'frame': count}  
                img = pickle.dumps(image)
                recv_Img.apply_async((data,img), serializer='pickle')
                success,image = vidcap.read() 
                count += 1
            self.video_map[proc]["total"]=count   
            fp.close()
            return "Thanks :)"
        else:
            return "Could not read any files:/"

    def get_imageinfo(self): 
        if request.method == 'POST':
            data=request.json
            frame_id = data['frame']
            frame_proc = data['proc']
            classes = data['classes']
            timestamp = data['timestamp']
            total = self.video_map[frame_proc]['total']
            self.video_map[frame_proc]["count"] += 1
            count = self.video_map[frame_proc]["count"]
            self.video_map[frame_proc]["timestamp"] += float(timestamp)
            lst=self.video_map[frame_proc]["classes"]
            self.video_map[frame_proc]["classes"] = self.mergeDict(lst,classes)
            if "person" in classes:
                if classes["person"]>self.max_persons:
                    print("Frame "+str(frame_id)+ ": " + str(classes["person"]) + " <person> detected") 
            if total == count:
                print("Processed frames: "+str(total))
                print("Average processing time per frame: "+str(int(self.video_map[frame_proc]["timestamp"]/count*1000))+"ms")
                print("Person objects detected: "+str(classes["person"]))
                print("Total classes detected: " + str(len(self.video_map[frame_proc]['classes'])))   
                k = Counter(self.video_map[frame_proc]["classes"])
                top = k.most_common(3)
                print("Top 3 objects detected: "+ self.printTop3(top))
        return ""

    def printTop3(self,lst):
        string=""
        for i in lst:
            string += i[0] + ", "
        string=string[:len(string)-2]
        return string
    #update the dicionary with the classes and it's frequency
    def mergeDict(self,dict1, dict2):
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = value + dict1[key]
        return dict3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", dest='max',help="maximum number of persons in a frame", default=10)
    parser.add_argument("-p", dest='port', type=int, help="HTTP port", default=5000)
    args = parser.parse_args()
    #pass the port and max number of persons to the Server 
    FlaskServer(args.port, args.max)

