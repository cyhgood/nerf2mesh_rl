import torch
import argparse
import math

from nerf.gui import NeRFGUI
from nerf.network import NeRFNetwork
from nerf.utils import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="recommended settings")
    '''
        -O =
    --fp16
    --preload
    --mark_untrained
    --random_image_batch
    --adaptive_num_rays
    --refine
    --mesh_visibility_culling
    '''    
    parser.add_argument('--workspace', type=str, default='workspace') #实验结果、checkpoint、网格保存目录
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0, help="training stage") #0 1
    parser.add_argument('--ckpt', type=str, default='latest') #加载最近一次checkpoint --scratch 从当前阶段重新开始
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")#使用半精度，降低现存占用

    ### testing options 测试选项
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='', help="nerfstudio compatible json file for camera trajactory")

    ### dataset options 数据集选项
    parser.add_argument('--data_format', type=str, default='nerf', choices=['nerf', 'colmap', 'dtu'])#数据格式
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'all'])
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")#将图像预加载到GPU
    parser.add_argument('--random_image_batch', action='store_true', help="randomly sample rays from all images per step in training stage 0, incompatible with enable_sparse_depth")
    parser.add_argument('--downscale', type=int, default=1, help="downscale training images")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1, help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--mesh', type=str, default='', help="template mesh for phase 2")
    parser.add_argument('--enable_cam_near_far', action='store_true', help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true', help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--enable_sparse_depth', action='store_true', help="use sparse depth from colmap pts3d, only valid if using --data_formt colmap")
    parser.add_argument('--enable_dense_depth', action='store_true', help="use dense depth from omnidepth calibrated to colmap pts3d, only valid if using --data_formt colmap")

    ### training options 训练选项
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")#nerf学习率
    parser.add_argument('--lr_vert', type=float, default=1e-4, help="initial learning rate for vert optimization")# stage 1网格定点偏移学习率
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")#标记没被相机观察到的空间区域
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=1000, help="training iters that only trains diffuse color for better initialization")
    parser.add_argument('--background', type=str, default='random', choices=['white', 'random'], help="training background mode")
    parser.add_argument('--enable_offset_nerf_grad', action='store_true', help="allow grad to pass through nerf to train vertices offsets in stage 1, only work for small meshes (e.g., synthetic dataset)")
    
    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true', help="adaptive num rays for more efficient training")#自动调整每次采样的射线数量
    parser.add_argument('--num_points', type=int, default=2 ** 18, help="target num points for each training step, only work with adaptive num_rays")

    # stage 0 regularizations stage0正则化
    parser.add_argument('--lambda_density', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale")#渲染权重 熵正则化
    parser.add_argument('--lambda_tv', type=float, default=1e-8, help="loss scale")
    parser.add_argument('--lambda_depth', type=float, default=0.1, help="loss scale")
    parser.add_argument('--lambda_specular', type=float, default=1e-5, help="loss scale") #镜面反射颜色L2正则化

    # stage 1 regularizations stage1正则化
    parser.add_argument('--wo_smooth', action='store_true', help="disable all smoothness regularizations")
    parser.add_argument('--lambda_lpips', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_offsets', type=float, default=0.1, help="loss scale")#顶点偏移L2正则化
    parser.add_argument('--lambda_sil', type=float, default=0, help='Stage 1 full-image silhouette L1 loss weight')
    parser.add_argument('--lambda_stage1_depth', type=float, default=0, help='Stage 1 valid surface depth L1 weight (default disabled)')
    parser.add_argument('--stage1_depth_manifest', default='', help='explicit training-view float NPY depth manifest')
    parser.add_argument('--lambda_lap', type=float, default=0.001, help="loss scale")#拉普拉斯平滑损失
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale")#法向一致性损失
    parser.add_argument('--lambda_edgelen', type=float, default=0, help="loss scale")

    # unused
    parser.add_argument('--contract', action='store_true', help="apply L-INF ray contraction as in mip-nerf, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--trainable_density_grid', action='store_true', help="update density_grid through loss functions, instead of directly update.")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--ind_dim', type=int, default=0, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=500, help="number of individual codes, should be larger than training dataset size")

    ### mesh options 网格选项
    # stage 0 粗网格提取
    parser.add_argument('--mcubes_reso', type=int, default=512, help="resolution for marching cubes")
    parser.add_argument('--env_reso', type=int, default=256, help="max layers (resolution) for env mesh")
    parser.add_argument('--decimate_target', type=int, default=3e5, help="decimate target for number of triangles, <=0 to disable")
    parser.add_argument('--mesh_visibility_culling', action='store_true', help="cull mesh faces based on visibility in training dataset")
    parser.add_argument('--visibility_mask_dilation', type=int, default=5, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=8, help="mesh clean: min face count for isolated mesh")#删除面数较少的孤立组件
    parser.add_argument('--clean_min_d', type=int, default=5, help="mesh clean: min diameter for isolated mesh")

    # stage 1
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--texture_size', type=int, default=4096, help="exported texture resolution")
    parser.add_argument('--refine', action='store_true', help="track face error and do subdivision")#开启自适应网格细化
    parser.add_argument('--refine_mode', choices=['original', 'local'], default='original', help='stage 1 topology refinement mode')
    parser.add_argument('--refine_experiment_steps', type=int, default=0, help='run a paired local refinement experiment for this many optimization steps')
    parser.add_argument('--refine_experiment_views', type=int, default=5, help='number of fixed validation views for the refinement experiment')
    parser.add_argument('--refine_selection_views', type=int, nargs='+', default=None, help='zero-based validation view indices for candidate selection')
    parser.add_argument('--refine_evaluation_views', type=int, nargs='+', default=None, help='zero-based validation view indices for evaluation')
    parser.add_argument('--refine_allow_view_overlap', action='store_true', help='allow selection and evaluation view indices to overlap')
    parser.add_argument('--refine_compare_random', action='store_true', help='compare high-error and random selection with matched face budgets')
    parser.add_argument('--refine_face_budget', type=int, default=0, help='maximum actual added faces for each comparison branch')
    parser.add_argument('--refine_random_seeds', type=int, nargs='+', default=[0], help='random selection seeds for repeated comparisons')
    parser.add_argument('--refine_rl', action='store_true', help='run the Stage 1 policy-gradient refinement experiment')
    parser.add_argument('--refine_rl_rounds', type=int, default=1, help='policy sampling and update rounds')
    parser.add_argument('--refine_rl_reward_views', type=int, nargs='+', default=None, help='validation view indices for training rewards; defaults to selection views')
    parser.add_argument('--refine_rl_lr', type=float, default=0.001, help='policy optimizer learning rate')
    parser.add_argument('--refine_rl_checkpoint', type=str, default='', help='load policy checkpoint')
    parser.add_argument('--refine_rl_eval', action='store_true', help='evaluate a loaded policy without updates')
    parser.add_argument('--refine_rl_eval_mode', choices=['rank', 'threshold'], default='rank', help='rank all visible faces or request only faces with sigmoid probability >= 0.5')
    parser.add_argument('--refine_multiround', action='store_true', help='run consecutive topology updates with face lineage rewards')
    parser.add_argument('--refine_observation_graph', action='store_true', help='record sparse face graphs in multi-round training and evaluation; policy stays MLP')
    parser.add_argument('--refine_graph_self_loops', action='store_true', help='add explicitly marked self loops to recorded observation graphs')
    parser.add_argument('--refine_multiround_rounds', type=int, default=3, help='maximum topology updates in this invocation')
    parser.add_argument('--refine_multiround_horizon', type=int, default=3, help='number of updates per truncated return segment')
    parser.add_argument('--refine_silhouette_radius', type=float, default=8.0, help='maximum pixel distance for assigning uncovered target silhouette to a face')
    parser.add_argument('--refine_multiround_gamma', type=float, default=0.9, help='lineage return discount')
    parser.add_argument('--refine_multiround_sil_weight', type=float, default=1.0, help='silhouette reward weight')
    parser.add_argument('--refine_multiround_rgb_weight', type=float, default=1.0, help='RGB reward weight')
    parser.add_argument('--refine_multiround_face_cost', type=float, default=0.001, help='cost per added face')
    parser.add_argument('--refine_advantage_baseline', choices=['none', 'mean'], default='mean', help='baseline applied to considered valid face returns')
    parser.add_argument('--refine_advantage_scale', choices=['none', 'std'], default='std', help='scale considered valid face advantages by their standard deviation')
    parser.add_argument('--refine_advantage_clip', type=float, default=5.0, help='clip absolute processed advantages; 0 disables clipping')
    parser.add_argument('--refine_candidate_fraction', type=float, default=0.5, help='fraction of valid faces selected by area')
    parser.add_argument('--refine_min_face_area', type=float, default=1e-12, help='minimum valid triangle area')
    parser.add_argument('--refine_gradient_ema_beta', type=float, default=0.9, help='vertex gradient EMA decay')
    parser.add_argument('--refine_gradient_ema_warmup_steps', type=int, default=0, help='optimization steps used to initialize vertex gradient EMA before the first refinement')
    parser.add_argument('--refine_reinitialize_legacy_policy', action='store_true', help='reinitialize incompatible completed legacy policy and vertex EMA')
    parser.add_argument('--refine_multiround_resume', type=str, default='', help='resume trajectory checkpoint')
    parser.add_argument('--refine_multibranch_eval', action='store_true', help='compare RL, high-error and random over consecutive refinement rounds')
    parser.add_argument('--refine_multibranch_policy', type=str, default='', help='multi-round trajectory.pt containing the fixed RL policy')
    parser.add_argument('--refine_multibranch_rounds', type=int, default=3, help='number of consecutive evaluation rounds')
    parser.add_argument('--refine_multibranch_total_budget', type=int, default=0, help='maximum total added faces; 0 means rounds times per-round budget')
    parser.add_argument('--refine_experiment_output', type=str, default='', help='output directory for the refinement experiment')
    parser.add_argument("--refine_steps_ratio", type=float, action="append", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--refine_size', type=float, default=0.01, help="refine trig length")
    parser.add_argument('--refine_decimate_ratio', type=float, default=0.1, help="refine decimate ratio")
    parser.add_argument('--refine_remesh_size', type=float, default=0.02, help="remesh trig length")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    if not math.isfinite(opt.lambda_stage1_depth) or opt.lambda_stage1_depth < 0:
        parser.error('--lambda_stage1_depth must be finite and nonnegative')
    if opt.lambda_stage1_depth > 0 and (opt.stage != 1 or not opt.stage1_depth_manifest):
        parser.error('Stage 1 depth requires --stage 1 and --stage1_depth_manifest')

    opt.cuda_ray = True

    if opt.O:
        opt.fp16 = True
        opt.preload = True
        opt.mark_untrained = True
        opt.random_image_batch = True
        opt.mesh_visibility_culling = True
        opt.adaptive_num_rays = True
        opt.refine = True
    
    if opt.contract:
        # mark untrained is not very correct in contraction mode...
        opt.mark_untrained = False
    
    # best rendering quality at the sacrifice of mesh quality
    if opt.wo_smooth:
        opt.lambda_offsets = 0
        opt.lambda_lap = 0
        opt.lambda_normal = 0
    
    if opt.enable_sparse_depth:
        print(f'[WARN] disable random image batch when depth supervision is used!')
        opt.random_image_batch = False
    
    if opt.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # 'nerf
        from nerf.provider import NeRFDataset
    
    # convert ratio to steps
    opt.refine_steps = [int(round(x * opt.iters)) for x in opt.refine_steps_ratio]

    seed_everything(opt.seed)

    model = NeRFNetwork(opt)
    
    # criterion = torch.nn.MSELoss(reduction='none')
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
            
            if not opt.test_no_mesh:
                if opt.stage == 1:
                    trainer.export_stage1(resolution=opt.texture_size)
                else:
                    # need train loader to get camera poses for visibility test
                    if opt.mesh_visibility_culling:
                        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
                    trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
        
    else:
        
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // 50) # save ~50 times during the training
        eval_interval = max(1, max_epoch // 10)
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        if opt.ind_dim > 0:
            assert len(train_loader) < opt.ind_num, f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to at least this number!"

        # colmap can estimate a more compact AABB
        if opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        # scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.iters // 2, opt.iters * 3 // 4, opt.iters * 9 // 10], gamma=0.33)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95 if opt.stage == 0 else None, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval, save_interval=save_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

            if opt.refine_experiment_steps:
                from nerf.refinement_experiment import run_refinement_experiment
                run_refinement_experiment(trainer, train_loader._data, valid_loader._data)
                raise SystemExit(0)

            trainer.metrics = [PSNRMeter(),]
            trainer.train(train_loader, valid_loader, max_epoch)
            
            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            if opt.stage == 1:
                trainer.export_stage1(resolution=opt.texture_size)
            else:
                trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
