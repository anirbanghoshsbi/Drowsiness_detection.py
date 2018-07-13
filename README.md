# Drowsiness Detection
(Implementation based on Dr. Adrian Rosebrock tutorial)

This code can be used to detect if the driver is drowsy at the wheels. If so then the program sounds an alert.

## Application 
Prevent accident from driver falling asleep at wheels by alerting them.

### The algorithm 

The algorithm hinges on two important computer vision techniques
a) facial landmark detection
b)Eye aspect ratio.

_Facial landmark prediction_ is the process of localizing key facial structures on a face, including the eyes, eyebrows, nose, mouth, and jawline.Specifically, in the context of drowsiness detection, we only needed the eye regions.

The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

The facial landmark detector is good for detecting mouth , left eyebrow , right eyebrow , right eye , left eye , nose , jawline. The indeces for these facial landmarks in the dlib facial landmark detector is as given below.

The mouth can be accessed through points [48, 68].
The right eyebrow through points [17, 22].
The left eyebrow through points [22, 27].
The right eye using [36, 42].
The left eye with [42, 48].
The nose using [27, 35].
And the jaw via [0, 17].

We apply the facial landmark detector and extract the eye region.Once we have the eye region with us then we compute the eye aspect ratio to determine if the eye is open or closed.If the algorithm detects the eye was xlosed for long enough time it sounds a alert.

Each eye is denoted by a 6 (x,y) co-ordinates the eye aspect ratio (EAR) is calculated using the formula :

EAR = (||p2-p6|| + ||p3-p5||)/(2*||p1-p4||) where p1 to p6 are the various key facial points as per the facial land mark detector of dlib.

```
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5]) # python indexing starts at 0 instead of 1.
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear
	```
if the Eye Aspect Ratio is low for longer than the threshold value of 25 seconds then it is assumed that the driver has slept on
the wheels , and alarm is araised.
	

