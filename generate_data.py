import os
import sys
import click
import cv2 as cv
import pathlib

from threading import Lock
from multiprocessing.pool import ThreadPool
from sys import platform
from collections import deque

try:
    # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('openpose/build/python/');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

except Exception as e:
    print(e)
    sys.exit(-1)

def set_params(): 
    params = dict()
    params["model_folder"] = "openpose/models/"
    params["face"] = True
    params["hand"] = True
    params["disable_blending"] = False
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

wrapper = set_params()


@click.command()
@click.option("--file", type=click.Path(exists=True), required=True)
@click.option("--directory", type=str, default="data")
@click.option("--save/--no-save", default=True)
def cli(file, directory, save):
    generate_data(file, directory, will_save=save)

def process_frame(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    wrapper.emplaceAndPop([datum])
    return frame, datum.cvOutputData
    

def generate_data(file, directory, will_save=True):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    cap = cv.VideoCapture(file)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    
    thread_num = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_num)
    pending = deque()
    frame_count = 0
    lock = Lock()
    print("Entering main Loop.")

    while cap.isOpened():
        while len(pending) > 0 and pending[0].ready():
            orig, new = pending.popleft().get()
            with lock:
                cv.imwrite(f'data/source_{frame_count}.png', orig)
                cv.imwrite(f'data/target_{frame_count}.png', new)
                if frame_count+1 % 100 == 0:
                    print(f"Processed {frame_count} frames")
                frame_count+=1
        
        if len(pending) < thread_num:
            _ret, frame = cap.read()
            if not _ret:
                continue
            task = pool.apply_async(process_frame, (frame.copy(),))
            pending.append(task)
        
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    cli()

