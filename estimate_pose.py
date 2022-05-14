import os
import torch
import imageio
import numpy as np
import cv2
from datetime import datetime
from utils import config_parser, load_synth, find_POI, img2mse, rot_phi, to8b
from nerf_helpers import load_nerf
from render_helpers import render, get_rays
from inerf_helpers import camera_transf

def get_init_pose(start_pose, batch, target, H, W, focal,  chunk, render_kwargs):
    pose = start_pose
    poses = []
    rot = rot_phi(20/180.*np.pi)
    prev_loss = torch.Tensor([1])
    loss_avg = []

    for k in range(18):
        
        poses.append(pose)
        pose = torch.matmul(torch.Tensor(rot), pose)

        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o[batch[:, 1], batch[:, 0]]
        rays_d = rays_d[batch[:, 1], batch[:, 0]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, _, _, _ = render(H, W, focal, chunk=chunk, rays=batch_rays,
                                        verbose=k < 10, retraw=True,
                                        **render_kwargs)
        loss = img2mse(rgb, target)
        
        if k == 0: prev_loss = loss
        loss_avg.append(np.mean([loss.item(),prev_loss.item()]))
        prev_loss = loss
    
    loss_avg[0] = np.mean([loss_avg[0], prev_loss.item()])
    
    return poses[np.argmin(loss_avg)]

def get_batch(batch_size, pool):
    if pool.shape[0] > batch_size: n = pool.shape[0]-1
    rand_inds = np.random.choice(n, size=batch_size, replace=False)
    return pool[rand_inds]

def run(device):

    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    experiment = args.experiment
    output_dir = args.output_dir
    model_name = args.model_name
    obj_name = args.obj_name
    obs_img_num = args.obs_img_num
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    dil_iter = args.dil_iter
    sampling_strategy = args.sampling_strategy
    feat_det = args.feat_det
    delta_phi, delta_theta, delta_psi = args.delta_phi, args.delta_theta, args.delta_psi
    delta_tx, delta_ty, delta_tz = args.delta_tx, args.delta_ty, args.delta_tz
    logging = args.logging
    overlay = args.overlay
    data_dir = args.data_dir
    white_bkgd = args.white_bkgd
    half_res = args.half_res

    lrate = 0.01

    # Load and pre-process the observed image
    # obs_img - rgb image with elements in range 0...255
    obs_img, hwf, start_pose, obs_img_pose = load_synth(data_dir, obj_name, obs_img_num,
                                            half_res, white_bkgd, delta_phi, delta_theta, delta_psi,
                                            delta_tx, delta_ty, delta_tz)
    H, W, focal = hwf
    near, far = 2., 6.

    # find points of interest of the observed image
    POI = find_POI(obs_img, feat_det)  # xy pixel coordinates of points of interest (N x 2)
    obs_img = (np.array(obs_img) / 255.).astype(np.float32)

    # create meshgrid from the observed image
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                        dtype=int)

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros((H, W, ), dtype=np.uint8)
    interest_regions[POI[:,1], POI[:,0]] = 1
    I = dil_iter
    interest_regions = cv2.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI contains all points except of POI
    coords = coords.reshape(H * W, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)

    # Load NeRF Model
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    # Create pose transformation model
    start_pose = torch.Tensor(start_pose).to(device)
    cam_transf = camera_transf().to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

    # Calculate angles and translation of the observed image's pose
    phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)

    if logging==1:
        start = datetime.now()
        file_name = f'logs/{model_name}_{sampling_strategy}_{experiment}.txt'
        log_file = open(file_name, "a")

        reference = f'{start}\n\nreference phi, theta, psi, translation: {phi_ref}, {theta_ref}, {psi_ref}, {obs_img_pose[0,3]}, {obs_img_pose[1,3]}, {obs_img_pose[2,3]}'
        log_file.write(reference+'\n\n'+'phi,theta,psi,rot_x,rot_y,rot_z,phi_err,theta_err,psi_err\n')
        trans_errs = []
        rot_errs = []
        losses = []

    testsavedir = os.path.join(output_dir, experiment)
    os.makedirs(testsavedir, exist_ok=True)

    # For Experiment III. - get init pose
    # rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
    # batch = interest_regions[rand_inds]
    # trgt = obs_img[batch[:, 1], batch[:, 0]]
    # trgt = torch.Tensor(trgt).to(device)
    # start_pose = get_init_pose(start_pose, batch, trgt, H, W, focal, obs_img, testsavedir, overlay, args.chunk, render_kwargs)

    # imgs - array with images are used to create a video of optimization process
    if overlay: imgs = []

    for k in range(40):
        if sampling_strategy == 'random': pool = coords
        elif sampling_strategy == 'interest_points': pool = POI
        else: pool = interest_regions

        batch = get_batch(batch_size, pool)

        target_s = obs_img[batch[:, 1], batch[:, 0]]
        target_s = torch.Tensor(target_s).to(device)
        pose = cam_transf(start_pose)

        rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
        rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 1], batch[:, 0]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=k < 10, retraw=True,
                                        **render_kwargs)

        optimizer.zero_grad()
        loss = img2mse(rgb, target_s)
        loss.backward()
        optimizer.step()
        
        new_lrate = lrate * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (k + 1) % 20 == 0 or k == 0:
            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()

                # Rendered pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)

                # Observed vs. rendered pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error

                # Logging
                translation_error = abs(translation_ref - translation)
                trans_errs.append(translation_error)
                rot_errs.append(rot_error)
                losses.append(loss.item())

                log_row = f'{phi},{theta},{psi},{pose_dummy[0,3]},{pose_dummy[1,3]},{pose_dummy[2,3]},{phi_error},{theta_error},{psi_error}'
                log_file.write(log_row+'\n')

            if overlay:
                with torch.no_grad():
                    rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                    rgb = rgb.cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(obs_img)
                    filename = os.path.join(testsavedir, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

    if logging == 1:
        end = datetime.now()
        errors_log = f'\nErrors:\nrotation,{rot_errs}\ntranslation,{trans_errs}\nloss,\n{losses}\n\n{end}'
        log_file.write(errors_log)
        log_file.close()

    if overlay is True:
        imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format
