import cv2
import mediapipe as mp
import numpy as np
import math
import imutils

# My variables
MY_CAMERA_MODE = 'REAR'
WINDOW_SIZE = 4

CAMERA_MODE = { 'FRONT': 1, 'REAR': 0 }
RINGS = [ 'assets/ring3.png', 'assets/ring1.png', 'assets/ring2.png' ]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
ring_index = 0

# For webcam input:
cap = cv2.VideoCapture(CAMERA_MODE[MY_CAMERA_MODE])
ring_images = [cv2.cvtColor(cv2.imread('assets/ring1_b.png'), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread('assets/ring2_b.png'), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread('assets/ring3_b.png'), cv2.COLOR_BGR2RGB)]

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    ring_image = ring_images[ring_index]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
            
        # # Step 1 - To find points on the hand
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())

        # Step 2 - To find the ring (or middle) finger and mapping the right position
        point_middle_finger_mcp_x, point_middle_finger_mcp_y = None, None
        point_index_finger_mcp_x, point_index_finger_mcp_y = None, None
        point_middle_finger_pip_x, point_middle_finger_pip_y = None, None
        
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x *w), int(lm.y*h)
            if id ==5:
                point_index_finger_mcp_x, point_index_finger_mcp_y = int(lm.x *w), int(lm.y*h)
                # cv2.circle(image, (point_index_finger_mcp_x,point_index_finger_mcp_y), 3, (255,0,255), cv2.FILLED)
            if id ==9:
                point_middle_finger_mcp_x, point_middle_finger_mcp_y = int(lm.x *w), int(lm.y*h)
                # cv2.circle(image, (point_middle_finger_mcp_x,point_middle_finger_mcp_y), 3, (255,0,255), cv2.FILLED)
            if id ==10:
                point_middle_finger_pip_x, point_middle_finger_pip_y = int(lm.x *w), int(lm.y*h)
                # cv2.circle(image, (point_middle_finger_pip_x,point_middle_finger_pip_y), 3, (255,0,255), cv2.FILLED)

        if( not (point_middle_finger_pip_x == None or point_middle_finger_pip_y == None or point_middle_finger_mcp_x == None or point_middle_finger_mcp_y == None or point_index_finger_mcp_x == None or point_index_finger_mcp_y == None) ):

            # print( "point_middle_finger_pip", point_middle_finger_pip_x, point_middle_finger_pip_y )
            # print( "point_middle_finger_mcp", point_middle_finger_mcp_x, point_middle_finger_mcp_y )
            # print( "point_index_finger_mcp", point_index_finger_mcp_x, point_index_finger_mcp_y )

            # To find the midpoint or target point
            cx, cy = int((point_middle_finger_pip_x + point_middle_finger_mcp_x)/2), int((point_middle_finger_pip_y + point_middle_finger_mcp_y)/2)
            # print( "Midpoint", cx, cy )
            # cv2.circle(image, (cx,cy), 5, (255,0,255), cv2.FILLED)

            # To find the angle
            angle_radians = math.atan2(point_middle_finger_mcp_y - point_middle_finger_pip_y, point_middle_finger_mcp_x - point_middle_finger_pip_x)
            angle_degrees = math.degrees(angle_radians)
            # print( "Angle", angle_degrees )
            
            # To find the width
            width = (((point_middle_finger_mcp_x - point_index_finger_mcp_x)**2) + ((point_middle_finger_mcp_y - point_index_finger_mcp_y)**2))**0.5
            # print( "Width", int(width) )

            # To find the perpendicular
            perp = - point_middle_finger_mcp_y + point_middle_finger_pip_y, point_middle_finger_mcp_x - point_middle_finger_pip_x
            perp_ptA_x, perp_ptA_y = cx - point_middle_finger_pip_y + point_middle_finger_mcp_y, cy + point_middle_finger_pip_x -point_middle_finger_mcp_x
            perp_ptB_x, perp_ptB_y = cx - point_middle_finger_mcp_y + point_middle_finger_pip_y, cy + point_middle_finger_mcp_x -point_middle_finger_pip_x
            # cv2.circle(image, (perp_ptA_x,perp_ptA_y), 3, (255,0,255), cv2.FILLED)
            # cv2.circle(image, (perp_ptB_x,perp_ptB_y), 3, (255,0,255), cv2.FILLED)
            
            # To get the position of ring
            for left_side_index in range( 0, abs(perp_ptA_x - cx), 1 ):
                left_side_x = int( cx - left_side_index )
                left_side_y = int( ((perp_ptA_y * perp_ptB_x) - ( perp_ptB_y * perp_ptA_x )  - (left_side_x * (perp_ptA_y - perp_ptB_y)))/(perp_ptB_x -perp_ptA_x ) )
                distance = (((cx - left_side_x)**2) + ((cy - left_side_y)**2))**0.5
                if( distance > (width / 2) ):
                    # print( "distance", distance )
                    break
                # cv2.circle(image, (left_side_x,left_side_y), 1, (255,0,0), cv2.FILLED)

            for right_side_index in range( 0, abs(perp_ptB_x - cx), 1 ):
                right_side_x = int( cx + right_side_index )
                right_side_y = int( ((perp_ptA_y * perp_ptB_x) - ( perp_ptB_y * perp_ptA_x )  - (right_side_x * (perp_ptA_y - perp_ptB_y)))/(perp_ptB_x -perp_ptA_x ) )
                distance = (((cx - right_side_x)**2) + ((cy - right_side_y)**2))**0.5
                if( distance > (width / 2) ):
                    # print( "distance", distance )
                    break
                # cv2.circle(image, (right_side_x,right_side_y), 1, (255,0,0), cv2.FILLED)
            
            # Resize and rotate
            ring_image_calculation = image_resize(ring_image, width = int(width))
            ring_image_calculation = imutils.rotate_bound(ring_image_calculation, int(angle_degrees) - 90 )

            # Mask the values with black spots in ring_image
            ring_shape_height, ring_shape_width, ring_image_dimensions = ring_image_calculation.shape
            start_width = int(cx - (ring_shape_height/2))
            start_height = int(cy - (ring_shape_width/2))
            replace_patch = image[start_height:start_height+ring_shape_height, start_width:start_width+ring_shape_width]
            border_pixel = ring_image_calculation[0,0,:]
            ring_image_calculation = cv2.cvtColor(ring_image_calculation, cv2.COLOR_RGB2BGR)
            ring_image_calculation = np.where(ring_image_calculation != border_pixel, ring_image_calculation, replace_patch )

            # Replace particular image patch with the ring_image
            image[start_height:start_height+ring_shape_height, start_width:start_width+ring_shape_width] = ring_image_calculation[:,:,0:3]

            # print("-----------------------------------------------------------------------------------------")
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

    if cv2.waitKey(33) == ord('a'):
        ring_index = (ring_index - 1) % len(ring_images)
        print("pressed Left Arrow", ring_index)
    if cv2.waitKey(33) == ord('d'):
        ring_index = (ring_index + 1) % len(ring_images)
        print("pressed Right Arrow", ring_index)
        
cap.release()