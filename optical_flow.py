import cv2
import numpy as np

class LucasKanadeOpticalFlow:
    """
    Sparse Optical Flow using Lucas-Kanade method
    Saves output as video file instead of displaying
    """
    
    def __init__(self, video_path):
        
        # open video file
        self.cap = cv2.VideoCapture(video_path)
        
        # get frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # save output as video file
        self.out = cv2.VideoWriter(
            'output_sparse.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (self.width, self.height)
        )
        
        # parameters for goodFeaturesToTrack
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # parameters for Lucas-Kanade
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                     cv2.TERM_CRITERIA_COUNT,
                     10, 0.03)
        )
        
        # random colors for tracking points
        self.colors = np.random.randint(0, 255, (100, 3))
        
        # read first frame
        ret, first_frame = self.cap.read()
        
        # convert to grayscale
        self.old_gray = cv2.cvtColor(
            first_frame,
            cv2.COLOR_BGR2GRAY
        )
        
        # find initial points to track
        self.p0 = cv2.goodFeaturesToTrack(
            self.old_gray,
            mask=None,
            **self.feature_params
        )
        
        # create mask for drawing lines
        self.mask = np.zeros_like(first_frame)
        
        print("Initialized successfully!")
        print(f"Tracking {len(self.p0)} points")
        print(f"Video size: {self.width}x{self.height}")
    
    def process_frame(self, frame):
        
        # convert to grayscale
        frame_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
        
        # calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.old_gray,
            frame_gray,
            self.p0,
            None,
            **self.lk_params
        )
        
        if p1 is None or status is None:
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray,
                mask=None,
                **self.feature_params
            )
            return frame

        # keep only good points
        good_new = p1[status == 1]    
        good_old = self.p0[status == 1]
        
        # draw lines and circles
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # draw motion line
            self.mask = cv2.line(
                self.mask,
                (int(a), int(b)),
                (int(c), int(d)),
                self.colors[i % 100].tolist(),
                2
            )
            
            # draw circle at new position
            frame = cv2.circle(
                frame,
                (int(a), int(b)),
                5,
                self.colors[i % 100].tolist(),
                -1
            )
        
        # combine frame and mask
        output = cv2.add(frame, self.mask)
        
        # update for next frame
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        
        return output
    
    def refresh_points(self):
        # refresh tracking points
        self.p0 = cv2.goodFeaturesToTrack(
            self.old_gray,
            mask=None,
            **self.feature_params
        )
        self.mask = np.zeros_like(self.mask)
    
    def run(self):
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Video ended!")
                break
            
            # process frame
            output = self.process_frame(frame)
            
            # add frame counter text
            cv2.putText(
                output,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # save frame to output video
            self.out.write(output)
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # refresh points every 30 frames
            if frame_count % 30 == 0:
                self.refresh_points()
                print(f"Refreshed points at frame {frame_count}")
        
        # cleanup
        self.cap.release()
        self.out.release()
        print("Done! Saved to output_sparse.mp4!")


if __name__ == "__main__":
    tracker = LucasKanadeOpticalFlow("video.mp4")
    tracker.run()