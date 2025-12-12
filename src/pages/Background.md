# Background
## State-of-the-Art vs. Our Approach
Most existing tracking systems work directly in image space. They measure player positions in pixels and try to follow these pixel movements over time. This creates several problems when applied to broadcast soccer footage.
* Pixel movements do not match real world movements because the camera is constantly zooming and panning.
* Partial visibility, such as players being blocked by others, makes detections unreliable.
* Occlusions and fast changes in camera angle often cause identity switches or lost tracks.
* It is difficult for these systems to stay consistent over long periods of time because image space motion is not stable.
* These limitations show why tracking only in image space is not enough for broadcast video.

In contrast, instead of tracking players in pixels like other methods, we track them in world space. By using a camera homography to map image detections from the image plane into xy space, we can better track each player’s true position and velocity over time, making our method more robust under the conditions mentioned above. The world-space coordinates remain stable through camera motion, partial visibility, and reduces the overall amount of ID switches.
