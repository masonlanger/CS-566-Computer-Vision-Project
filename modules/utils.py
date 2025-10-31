# import torch

# def (detections, homographies):
#     T = len(detections)
#     for t in range(T):
        

#         one = detection.new_tensor(1.0).unsqueeze(0)
#         img_xyz = torch.cat([detection, one], dim=0)
#         world_xyz = H @ img_xyz
#         world_xyz[:2] /= world_xyz[2]
#     ...