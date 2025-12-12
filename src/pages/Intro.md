---
layout: ../layouts/Layout.astro
---
# Introduction
In professional soccer, accurate player tracking is very important for analysis, coaching, and fair broadcasting. But most high-quality tracking systems use expensive multi-camera setups or private datasets that many teams, researchers, and leagues cannot access. Broadcast soccer videos are easy to find, but they are much harder to use for tracking. Players become blocked by others, the camera moves quickly, and the view often changes from one angle to another. Because of this, many tracking methods lose players, mix up identities, or produce unstable location estimates.

Our project focuses on solving the main problem of tracking players in world space using only regular broadcast video, even when visibility is poor. We want to keep each player’s identity consistent across the entire match, even when they are hidden, blurred, or off-screen for short periods. This type of tracking is important because it allows us to calculate useful soccer information such as team formations, passing patterns, and expected goals. These metrics normally require precise player positions, which are not available in many leagues.

The motivation for this work comes from real needs in the sports world. Broadcasters want reliable tracking to support fair officiating, especially when rain, glare, or crowding make the game hard to see. Analysts and coaches rely on accurate location data to study performance. Fans now expect real-time information during games, such as player speed or heatmaps, which all depend on stable tracking.

By building a world space tracking system that learns motion patterns directly from video, our project aims to provide a practical and accessible solution that works with the footage that most people already have. Even small improvements can help with understanding player movement, judging offside situations, and improving the viewing experience for everyone.
