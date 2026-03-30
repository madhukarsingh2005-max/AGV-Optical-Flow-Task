import cv2
import numpy as np

class DenseOpticalFlow:
    """
    Dense Optical Flow using Farneback method
    Saves output as video file instead of displaying
    """
    
    def __init__(self, video_path):
        
        # store video path
        self.video_path = video_path
        
        # open video file
        self.cap = cv2.VideoCapture(video_path)
        
        # check if video opened successfully
        if not self.cap.isOpened():
            print("ERROR: Cannot open video file!")
            return
        
        print(f"Video opened: {self.cap.isOpened()}")
        print(f"Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
        # get frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video size: {self.width}x{self.height}")
        
        # save output as video file
        self.out = cv2.VideoWriter(
            'output_dense.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (self.width, self.height)
        )
        
        print("Dense Optical Flow initialized!")
    
    def process_frame(self, frame, old_gray):
        
        # convert to grayscale
        frame_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
        
        # calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray,
            frame_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
        # create HSV image
        hsv = np.zeros_like(frame, dtype=np.float32)
        hsv[..., 1] = 255
        
        # convert to magnitude and angle
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )
        
        # set color based on direction
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # normalize magnitude
        magnitude = cv2.normalize(magnitude, None, 30, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = magnitude
        
        # convert HSV to BGR
        hsv = np.uint8(hsv)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return output, frame_gray
    
    def run(self):
        
        frame_count = 0
        old_gray = None
        
        while True:
            ret, frame = self.cap.read()
            
            # check if frame is valid
            if not ret or frame is None:
                print("Video ended!")
                break
            
            # first frame - just save as old_gray
            if frame_count == 0:
                old_gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )
                frame_count += 1
                print(f"First frame captured!")
                continue
            
            # process frame
            output, old_gray = self.process_frame(frame, old_gray)
            
            # save frame to output video
            self.out.write(output)
            
            frame_count += 1

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")
            
        
        self.cap.release()
        self.out.release()
        print(f"Done! Processed {frame_count} frames!")
        print("Saved to output_dense.mp4!")


if __name__ == "__main__":
    tracker = DenseOpticalFlow("video.mp4")
    tracker.run()