---
layout: ../layouts/Layout.astro
---
# Limitations & Future Considerations
Although our world space tracking approach shows promise, it still has important limitations. The current system depends on accurate homography estimation, which is the process of mapping the broadcast camera view onto the soccer field. If the homography is incorrect because of camera motion, zoom, or poor field visibility, then the world space positions of the players become less reliable. As a result, tracking can drift or become unstable during parts of the match.

There are several directions for future work that can improve the system. One direction is learning the transition model directly from data instead of hand designing it. A learned model could better capture how players actually move and react during a game, which would make world space predictions more consistent. Another direction is improving data association by adding appearance information from the video. This would help the system keep the correct identity for each player, even when players look similar or when they are partially hidden.

Together, these future improvements would make the tracking stronger, more stable, and more useful for real time soccer analytics.
