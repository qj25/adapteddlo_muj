import os
import cv2
import numpy as np
import pickle

from adapteddlo_muj.utils.finddepth import split_lines
from adapteddlo_muj.utils.argparse_utils import d2p_parse

parser = d2p_parse()
args = parser.parse_args()
wire_color = args.wirecolor
new_ptselect = bool(args.newptselect)
testid = args.testid

# Global list to store points
points = []

# Mouse callback function to capture points
def select_point(event, x, y, flags, param):
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Mark the point on the image
        if len(points)>1:
            cv2.line(img, points[-2], points[-1], (0, 0, 255), 1) 
        cv2.imshow("Image", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        points.pop()
        print("Undo-ed last")

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname, cv2.WND_PROP_FULLSCREEN)        # Create a named window
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

# load pickle of pos
# realdata_picklename = wire_color + '0_data.pickle'
realdata_picklename = wire_color + testid + '_data.pickle'
realdata_picklename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "adapteddlo_muj/data/dlo_muj_real/" + realdata_picklename
)
# Load the image
image_name = wire_color + '0.jpg'  # Replace with the path to your image
# image_name = wire_color + testid + '.jpg'  # Replace with the path to your image
image_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'adapteddlo_muj/data/dlo_muj_real/' + image_name
)

clamp_realdist = 0.12
total_reallen = 1.5
r_pieces = 50
# r_pieces -= 1 # to account for the 2 half-pieces at the end at beginning for the sim 
if new_ptselect:
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Unable to load image.")
        exit()

    # Set up the window and mouse callback
    scale_factor = 1.0
    img_resized = cv2.resize(img,(
        int(img.shape[1]*scale_factor),
        int(img.shape[0]*scale_factor)
    ))
    # cv2.imshow("Image", img )
    showInMovedWindow("Image", img, 0, 0)
    cv2.setMouseCallback("Image", select_point)

    # Wait for the user to press 'q' to exit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Convert the list of points to a numpy array
    points_array = np.array(points)
    print("Selected points:", points_array)

    # Clean up and close all windows
    cv2.destroyAllWindows()

    # Optionally save the points to a file
    # np.savetxt('selected_points.txt', points_array, fmt='%d')

    # first and last points are the clamp ends, and start and end of the wire
    # Using a 3x telephoto camera on Samsung S22 Ultra, minimal distortion

    clamp_imgdist = np.linalg.norm(points_array[0] - points_array[-1])
    clamp_img2real_dist = clamp_realdist/clamp_imgdist

    total_imglen = 0.0
    for i in range(len(points_array)-1):
        total_imglen += np.linalg.norm(points_array[i]-points_array[i+1])
    wire_img2real_dist = total_reallen/total_imglen
    seg_len = total_imglen/r_pieces

    # make start of rope the (0.0,0.0) point
    start_pt = points_array[0].copy()
    points_array -= start_pt
    newsplit_lines = split_lines(points_array,seg_len,0.0)
    newsplit_lines_real = newsplit_lines * wire_img2real_dist

    print("Check difference in img2real")
    print(f"-clamp_img2real_dist = {clamp_img2real_dist}")
    print(f"-wire_img2real_dist = {wire_img2real_dist}")
    print(f"newsplit_lines = {newsplit_lines_real}")

    pickle_data = [newsplit_lines_real,[newsplit_lines,start_pt]]
    with open(realdata_picklename, 'wb') as f:
        pickle.dump(pickle_data,f)

else:
    with open(realdata_picklename, 'rb') as f:
        newsplit_lines_real, [newsplit_lines, start_pt] = pickle.load(f)
    total_imglen = 0.0
    for i in range(len(newsplit_lines)-1):
        total_imglen += np.linalg.norm(newsplit_lines[i]-newsplit_lines[i+1])
    wire_img2real_dist = total_reallen/total_imglen
    seg_len = total_imglen/r_pieces
    newsplit_lines = split_lines(newsplit_lines,seg_len,0.0)
    newsplit_lines_real = newsplit_lines * wire_img2real_dist
    # # for updating the current pickle
    # pickle_data = [newsplit_lines_real,[newsplit_lines,start_pt]]
    # with open(realdata_picklename, 'wb') as f:
    #     pickle.dump(pickle_data,f)

nslines_int = (newsplit_lines+start_pt).astype('int')
# show again to test
img2 = cv2.imread(image_path)
showInMovedWindow("Image2", img2, 0, 0)
for i in range(len(nslines_int)):
    cv2.circle(img2, nslines_int[i], 5, (255, 0, 0), -1)  # Mark the point on the image
    if i>0:
        cv2.line(img2, nslines_int[i], nslines_int[i-1], (255, 0, 0), 1) 
    cv2.imshow("Image2", img2)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
