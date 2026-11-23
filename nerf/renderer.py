import os
import cv2
import math
import json
import tqdm
import mcubes
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
import nvdiffrast.torch as dr

import xatlas
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

from .utils import custom_meshgrid, plot_pointcloud
from meshutils import *

# import torch_scatter
TORCH_SCATTER = None # lazy import

#nerf/renderer.py 是 NeRF2Mesh 中负责“渲染、密度计算和网格导出”的核心模块，
# 相当于连接 NeRFNetwork 与训练流程的中间层。

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

class NeRFRenderer(nn.Module):#
    # 初始化渲染器，读取bound，网格大小、采样数据参数等配置，并建立密度网格
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = 2
        else:
            self.bound = opt.bound
        
        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.grid_size = opt.grid_size
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # individual codes
        self.individual_num = opt.ind_num
        self.individual_dim = opt.ind_dim

        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(self.individual_num, self.individual_dim) * 0.1) 
        else:
            self.individual_codes = None

        # extra state for cuda raymarching
        self.cuda_ray = opt.cuda_ray
        assert self.cuda_ray

        # density grid 建立密度网格
        if not self.opt.trainable_density_grid:
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            self.register_buffer('density_grid', density_grid)
        else:
            self.density_grid = nn.Parameter(torch.zeros([self.cascade, self.grid_size ** 3])) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0
        
        # for second phase training 第二阶段训练

        if self.opt.stage == 1:
            
            if self.opt.gui:
                self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            else:
                self.glctx = dr.RasterizeGLContext(output_db=False) # will crash if using GUI...

            # sequentially load cascaded meshes
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            for cas in range(self.cascade):
                
                _updated_mesh_path = os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}_updated.ply') if self.opt.mesh == '' else self.opt.mesh
                if os.path.exists(_updated_mesh_path) and self.opt.ckpt != 'scratch':
                    mesh = trimesh.load(_updated_mesh_path, force='mesh', skip_material=True, process=False)
                else: # base (not updated)
                    mesh = trimesh.load(os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}.ply'), force='mesh', skip_material=True, process=False)
                print(f'[INFO] loaded cascade {cas} mesh: {mesh.vertices.shape}, {mesh.faces.shape}')

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

            vertices = np.concatenate(vertices, axis=0)
            triangles = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

            # must put to cuda manually, we don't want these things in the model as buffers...
            self.vertices = torch.from_numpy(vertices).float().cuda() # [N, 3]
            self.triangles = torch.from_numpy(triangles).int().cuda()

            # learnable offsets for mesh vertex 网格顶点的可学习偏移
            self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

            # accumulate error for mesh face 计算网格面的误差
            self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
            self.triangles_errors_id = None
            # Maps each current face to its face index before the latest local update.
            self.parent_face_id = None
            
        else:
            self.glctx = None

    def get_params(self, lr):
        params = []

        if self.individual_codes is not None:
            params.append({'params': self.individual_codes, 'lr': self.opt.lr * 0.1, 'weight_decay': 0})

        if self.opt.trainable_density_grid:
            params.append({'params': self.density_grid, 'lr': self.opt.lr, 'weight_decay': 0})

        if self.glctx is not None:
            params.append({'params': self.vertices_offsets, 'lr': self.opt.lr_vert, 'weight_decay': 0})

        return params
    #	综合计算空间点的密度和颜色，供训练或渲染调用
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    #	输入三维坐标 x，调用 NeRFNetwork 计算密度 sigma 和几何特征
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()
      
    #	清空密度网格、占用区域等辅助状态，重新开始统计
    def reset_extra_state(self):
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    def update_aabb(self, aabb):
        # aabb: tensor of [6]
        if not torch.is_tensor(aabb):
            aabb = torch.from_numpy(aabb).float()
        self.aabb_train = aabb.clamp(-self.real_bound, self.real_bound).to(self.aabb_train.device)
        self.aabb_infer = self.aabb_train.clone()
        print(f'[INFO] update_aabb: {self.aabb_train.cpu().numpy().tolist()}')

    @torch.no_grad()
    def refine_and_decimate(self, export_mesh=True, candidate_order=None, max_added_faces=None): #自适应网格细化的核心函数
        #1.确保当前处于stage1
        assert self.opt.stage > 0
        #2.保存设备信息
        device = self.vertices.device

        #3.获取当前网格
        v = (self.vertices + self.vertices_offsets).detach().cpu().numpy()
        f = self.triangles.detach().cpu().numpy()

        #4.读取三角形误差和观察次数
        errors = self.triangles_errors.cpu().numpy() #errors[f]：三角形 f 的累计误差
        cnt = self.triangles_errors_cnt.cpu().numpy() #cnt[f]：三角形 f 被多少个像素观察到
        
        #5.计算平均三角形误差
        cnt_mask = cnt > 0
        errors[cnt_mask] = errors[cnt_mask] / cnt[cnt_mask]

        # only care about the inner mesh
        #6.只处理内层网络
        errors = errors[:self.f_cumsum[1]] #self.f_cumsum 用来记录不同 cascade 网格的面片边界
        #f_cumsum = [0, 50000, 70000] 表示mesh_0：三角形 [0, 50000) mesh_1：三角形 [50000, 70000)
        cnt_mask = cnt_mask[:self.f_cumsum[1]]

        # find a threshold to decide whether we perform subdivision / decimation.
        #7.计算细化和简化阈值 使用误差的百分位数划分三角形
        local_refine = getattr(self.opt, 'refine_mode', 'original') == 'local'
        if local_refine and not cnt_mask.any():
            thresh_refine = np.inf
        else:
            thresh_refine = np.percentile(errors[cnt_mask], 90) #表示误差排在前 10% 左右的阈值 errors > thresh_refine 需要细分
          
        if not local_refine:
            thresh_decimate = np.percentile(errors[cnt_mask], 50) #errors < thresh_decimate，可以简化

        #8.创建网格操作mask
        mask = np.zeros_like(errors) #初始时所有三角形都设置为0：不处理
        mask[(errors > thresh_refine) & cnt_mask] = 2 #高误差区域 2：细分
        if candidate_order is not None:
            if not local_refine:
                raise ValueError('Explicit candidates require local refinement')
            mask[:] = 0
            mask[candidate_order] = 2
        if not local_refine:
            mask[(errors < thresh_decimate) & cnt_mask] = 1 #低误差区域 1：简化

        #9.输出要处理的面片数量
        print(f'[INFO] faces to decimate {(mask == 1).sum()}, faces to refine {(mask == 2).sum()}')

        #一、场景范围不超过1 只有内层网格mesh0
        if self.bound <= 1:

            # mesh = trimesh.Trimesh(v, f, process=False)
            # mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_before_updated.ply'))

            #11.执行网格简化和细分 这个函数真正修改网格拓扑 
            #refine_decimate_ratio 0.1 低误差区域简化比例 refine_size 0.01 高误差区域细分的边长阈值
            #refine_remesh_size 0.02 简化区域重新网格化的目标边长
            if local_refine:
                v, f, parent_face_id, event = subdivide_longest_edges(v, f, mask == 2, return_event=True, candidate_order=candidate_order, max_added_faces=max_added_faces)
            else:
                v, f = decimate_and_refine_mesh(v, f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)
            
            # export
            #12.保存更新后的内层网格
            mesh = trimesh.Trimesh(v, f, process=False)
            #创建新的网格对象 mesh_0_updated.ply 用于检查当前自适应细化后的网格
            if export_mesh:
                mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_updated.ply'))
            #13.重新读取网格数据
            v, f = mesh.vertices, mesh.faces

            # fix counters
            #14.修正累计计数 细分和简化之后，顶点数和三角形数都会改变，因此需要更新
            self.v_cumsum[1:] += v.shape[0] - self.v_cumsum[1]
            self.f_cumsum[1:] += f.shape[0] - self.f_cumsum[1]
        
        #二、场景范围大于1 此时不能直接对整个网格一起修改，而是要按照 cascade 分开处理
        else:
            #15.创建新的网格列表 用于保存每个 cascade 更新后的：顶点、三角形，顶点累计数，面片累计数
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            parent_faces = []

            #16.遍历每个cascade 
            for cas in range(self.cascade):

                #17.切分当前cascade网格
                #提取当前cascade的顶点 
                cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
                #提取当前 cascade 的三角形，并减去顶点偏移量
                #为什么减去self.v_cumsum[cas]
                #因为整个网格的三角形索引是全局索引，而 cur_v 是局部顶点数组，需要转换为局部索引
                cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]

                #18.只细化内层网格
                if cas == 0:
                    if local_refine:
                        cur_v, cur_f, parent_face_id, event = subdivide_longest_edges(cur_v, cur_f, mask == 2, return_event=True)
                    else:
                        cur_v, cur_f = decimate_and_refine_mesh(cur_v, cur_f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)
                elif local_refine:
                    parent_face_id = np.arange(len(cur_f), dtype=np.int64)

                if local_refine:
                    parent_faces.append(parent_face_id + self.f_cumsum[cas])

                #19.保存各cascade网格
                mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
                if export_mesh:
                    mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', f'mesh_{cas}_updated.ply'))

                #20.将局部网格重新合并 当前网格的面片索引原本是局部索引，需要加上前面网格的顶点数量，转换成全局索引
                vertices.append(mesh.vertices)
                #eg.当前局部面片：[0, 1, 2] 前面已有顶点数：10000 全局面片：[10000, 10001, 10002]
                triangles.append(mesh.faces + v_cumsum[-1])

                #21.更新累计数量  重新记录每个 cascade 的顶点和面片范围
                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
            #22.合并所有cascade 将所有 cascade 网格合并成一个完整网格
            v = np.concatenate(vertices, axis=0)
            f = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)
            if local_refine:
                parent_face_id = np.concatenate(parent_faces)

        #三、重新放回GPU
        #23.重新更新顶点和三角形 将更新后的顶点NumPy CPU数组 转换回GPU Float32 Tensor
        self.vertices = torch.from_numpy(v).float().contiguous().to(device) # [N, 3]
        self.triangles = torch.from_numpy(f).int().contiguous().to(device)
        self.parent_face_id = torch.from_numpy(parent_face_id).to(device) if local_refine else None
        self.refinement_event = event if local_refine else None
        #24.重置顶点偏移量 因为网格拓扑已经发生改变 新的 vertices_offsets = 0 之后 Stage 1 会继续学习新的顶点偏移
        self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

        #25.清空三角形误差 重新创建数组
        self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        #26.输出新网格规模
        print(f'[INFO] update stage1 mesh: {self.vertices.shape}, {self.triangles.shape}')


    @torch.no_grad()
    #	Stage 1 训练结束后，导出最终网格、UV 坐标和纹理
    def export_stage1(self, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)
        #二、检查当前是否为Stage1
        assert self.opt.stage > 0
        device = self.vertices.device

        #三、定义单个网格的导出函数 ssaa超采样倍数 mesh_0.obj
        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor
            #四、转换为NumPy
            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            #五、使用xatlas进行UV展开
            #1.创建UV Atlas 将三维网格加入 xatlas UV 展开的目的是将三维表面映射到二维纹理平面
            #(x,y,z)->(u,v)这样才能将三维网格表面颜色保存为二维纹理图
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            #2.设置展开参数
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0 #max_iterations = 0 表示关闭chart合并迭代，主要目的是加快 UV 展开速度
            pack_options = xatlas.PackOptions() #PackOptions 用于将多个UV chart尽可能紧凑地排布到纹理图中
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            
            #3.生成UV
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            #执行UV展开 vmapping UV顶点与原始顶点的映射关系
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]
            # ft_np UV顶点坐标，范围通常为 [0,1] 	vt_np 每个三角形对应的UV顶点索引

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]
            #六、将UV坐标放到GPU ，将 UV 坐标和 UV 面片索引转换为 GPU Tensor，供 nvdiffrast 使用
            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            #七、在UV空间中进行光栅化
            #1.将UV坐标转换到裁剪空间 [0,1]->[-1,1]
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            #2.添加z和w坐标 [u, v]扩展为四维齐次坐标[u, v, 0, 1] 最终uv: [N_uv, 4]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]
            #这里的 UV 坐标被当作一个二维平面上的“网格”进行光栅化

            #3.设置纹理分辨率
            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            #4.光栅化UV三角形 这一步将三维网格的 UV 展开结果光栅化到二维纹理平面 rast: [1, h, w, 4]
            #每个纹理像素可以知道：是否落在三角形内；对应重心坐标；对应三角形
            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            
            #八、从UV像素恢复三维表面坐标
            #1.插值三维坐标 这里使用 UV 光栅化得到的重心坐标，对三维顶点进行插值
            #Xuv=b1v1+b2v2+b3v3 b1b2b3:UV空间中的重心坐标 v1v2v3:三角形三个顶点
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            #2.获取纹理掩码 给所有顶点设置值 1，再通过插值得到纹理掩码
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            #九、查询NeRF特征
            #1.创建六通道特征图 每个纹理像素对应6个特征值 之后分为两张三通道图：前三通道-feat0，后三通道-feat1
            feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

            if mask.any():
                #2.提取有效三维点 只保留落在网格表面的三维点
                xyzs = xyzs[mask] # [M, 3]

                # batched inference to avoid OOM
                #3.分块查询颜色特征
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    #4.调用geo_feat() 根据三维位置预测 6 通道颜色特征
                    #三维表面点-颜色哈希编码-color_net->6维特征 前三漫反射颜色，后三高光网络使用的特征
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        all_feats.append(self.geo_feat(xyzs[head:tail]).float())
                    head += 640000 #每次最多处理640000个点
                #5.将特征写回UV图 将有效网格区域的 6 通道特征写入纹理图
                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1) # 6 channels 然后恢复二维结构 feats: [h, w, 6]
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            #十、量化为8位纹理图
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8) [0,255]

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            #十一、填充UV边界空间
            #1.膨胀得到待填充区域
            inpaint_region = binary_dilation(mask, iterations=3)#对有效区域进行膨胀，得到网格边界附近的区域
            inpaint_region[mask] = 0 #去掉原本已经有效的区域，只保留外部附近的空白区域

            #2.获取有效区域边界 经过腐蚀后，只保留有效区域靠近边界的一圈像素 这些边界像素作为纹理填充的搜索源
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0

            #3.获取坐标
            search_coords = np.stack(np.nonzero(search_region), axis=-1) #可用于搜索的有效纹理坐标
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1) #需要填充的空白坐标

            #4.最近邻填充 建立 KD-tree 为每个空白像素寻找最近的有效边界像素
            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
            #使用最近有效像素的特征填充空白区域 这不是严格意义上的神经网络抗锯齿，而是利用最近邻搜索减少纹理边缘的黑缝和采样空洞

            # do ssaa after the NN search, in numpy
            #十二、拆分两类特征纹理  前 3 个通道保存为 feat0 通常表示基础颜色或漫反射特征
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR) # albedo
            #后 3 个通道保存为 feat1 通常用于后续高光网络或可见性相关计算
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR) # visibility features

            #十三、缩小超采样纹理 如果之前使用了超采样，则使用双线性插值将纹理缩小到目标分辨率。
            #注意，这一步发生在最近邻填充之后
            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            #十四、保存两张纹理图
            #feat0：基础颜色/漫反射特征；
            cv2.imwrite(os.path.join(path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            #feat1：用于高光或可见性计算的特征。
            cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])

            # save obj (v, vt, f /)
            #十五、创建OBJ和MTL文件
            #1.设置文件路径
            obj_file = os.path.join(path, f'mesh_{cas}.obj')
            mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:

                #2.写入材质文件引用 表示OBJ使用对应的MTL文件
                fp.write(f'mtllib mesh_{cas}.mtl \n')
                
                print(f'[INFO] writing vertices {v_np.shape}')
                #3.写入三维顶点 OBJ中 v x y z
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                #4.写入UV坐标 OBJ中 vt u v
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 
                    #1 - v[1] 是因为图像坐标和 OBJ UV 坐标的纵轴方向相反：UV 坐标通常从左下角开始，因此需要翻转 V 坐标

                print(f'[INFO] writing faces {f_np.shape}')
                #5.写入三角形面片
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")
                    #OBJ 索引从 1 开始，而 NumPy 索引从 0 开始 每个面片格式：顶点索引/UV索引
                    #eg.f 1/1 2/2 3/3 表示：三维顶点1对应UV顶点1...

            with open(mtl_file, "w") as fp:
                #十六、写入MTL材质文件 defaultMat
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n') #环境光颜色
                fp.write(f'Kd 1 1 1 \n') #漫反射颜色
                fp.write(f'Ks 0 0 0 \n') #镜面反射颜色
                fp.write(f'Tr 1 \n') #透明度
                fp.write(f'illum 1 \n') #光照模式
                fp.write(f'Ns 0 \n') #高光系数
                fp.write(f'map_Kd feat0_{cas}.png \n')
                
                #mesh_0.obj + mesh_0.mtl + feat0_0.png
        #十七、提取优化后的网格
        v = (self.vertices + self.vertices_offsets).detach()
        f = self.triangles.detach()
        #十八、遍历不同cascade
        for cas in range(self.cascade):
            #1.提取当前cascade的网格
            #取出当前cascade的顶点
            cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
            #取出当前cascade的面片，并转换成局部顶点索引
            cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]
            #2.导出当前cascade
            _export_obj(cur_v, cur_f, h0, w0, self.opt.ssaa, cas)
            #完成当前mesh的：UV展开、UV光栅化、特征查询、纹理保存、OBJ/MTL保存

            # half the texture resolution for remote area.
            #3.远处区域降低纹理分辨率 对于远处的 cascade，纹理分辨率减半
            if h0 > 2048 and w0 > 2048:
                h0 //= 2
                w0 //= 2

        # save mlp as json
        #十九、导出高光MLP
        #1.获取高光网络参数
        params = dict(self.specular_net.named_parameters())

        mlp = {}
        #2.转换网络参数 将 PyTorch 参数转换为 NumPy，并进行转置
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f'[INFO] wrting MLP param {k}: {p_np.shape}')
            mlp[k] = p_np.tolist() #将参数转换成 JSON 可保存的列表
        #3.保存场景信息
        mlp['bound'] = self.bound
        mlp['cascade'] = self.cascade

        #4.写入mlp.json 这个文件用于在外部渲染器中加载 specular_net，从而恢复完整的 漫反射 + 高光
        mlp_file = os.path.join(path, f'mlp.json')
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)

    
    @torch.no_grad()
    #把 Stage 0 训练得到的 NeRF 密度场转换成初始三角网格，并保存
    #1.函数定义，decimate_target：网格简化后的目标三角形数量；dataset：训练数据，用于可见性剪裁；S：分块查询密度时的块大小
    #s=128，表示每次最多处理约128^3个空间点，避免一次性查询过多点造成显存不足；resolution密度场提取分辨率
    def export_stage0(self, save_path, resolution=None, decimate_target=1e5, dataset=None, S=128):

        # only for the inner mesh inside [-1, 1]
        #2.设置提取分辨率（--mcubes_reso=512)
        if resolution is None:
            resolution = self.grid_size #如果没有指定分辨率，就使用当前密度网格的分辨率。
        #3.设置计算设备
        device = self.density_bitfield.device
        
        #4.确定密度阈值，这里确定 Marching Cubes 使用的密度阈值
        #密度大于 density_thresh → 认为可能属于物体；密度小于 density_thresh → 认为属于空区域
        density_thresh = min(self.mean_density, self.density_thresh)#是为了避免阈值过高导致物体表面被截断或消失

        # sigmas = np.zeros([resolution] * 3, dtype=np.float32)
        #5.创建密度体
        sigmas = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)
        
        #一、从已有密度网格提取内层网格
        #6.分辨率等于网格分辨率，就不需要重新调用网络查询密度，可以直接使用已经保存的self.density_grid[0]
        if resolution == self.grid_size:
            # re-map from morton code to regular coords...
            #7.生成Morton索引：生成0-resolution^3-1的线性索引
            all_indices = torch.arange(resolution**3, device=device, dtype=torch.int)
            #8.Morton编码转回三维坐标(x,y,z)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            #9.将压缩密度网格还原成规则网格
            #这一步将 density_grid[0] 中按照 Morton 顺序存储的密度，重新放回规则的三维数组，这样 Marching Cubes 才能处理
            sigmas[tuple(all_coords.T)] = self.density_grid[0] #sigma[x,y,z]
        
        #二、重新查询密度场 就需要在规则三维网格中重新查询 NeRF 密度
        else:
            # query 11.在[-1,1]范围内生成采样坐标
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)
            #如果resolution=512，S=128，每个方向分成四小块，最终循环处理64个空间块
            #12.遍历每个空间块
            for xi, xs in enumerate(X): #遍历x方向的块
                for yi, ys in enumerate(Y): #遍历y方向的块
                    for zi, zs in enumerate(Z): #遍历z方向的块
                      #13.创建三维网格坐标 将一维坐标转换为三维坐标网格
                        #eg：xs: [x1, x2, ...]，ys: [y1, y2, ...]，zs: [z1, z2, ...]
                        #xx[i,j,k] = xi ，yy[i,j,k] = yj ，zz[i,j,k] = zk
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        
                        #14.拼接三维坐标 pts[当前块点数,3] 每一行代表一个三维空间点：[x,y,z]
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                        
                        #15.调用NeRF查询密度
                        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                            #val:[当前块x大小,当前块y大小,当前块z大小]
                            val = self.density(pts.to(device))['sigma'].reshape(len(xs), len(ys), len(zs)) # [S, 1] --> [x, y, z]
                        #16.将局部密度放回完整密度体
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
            
            #三、利用density_grid过滤无效区域
            # use the density_grid as a baseline mask (also excluding untrained regions)
            #17.创建基准mask 创建一个与原始密度网格大小相同的三维掩码
            mask = torch.zeros([self.grid_size] * 3, dtype=torch.float32, device=device)
            #18.还原原始occupancy网格
            all_indices = torch.arange(self.grid_size**3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).long()
            #将第 0 层密度网格重新转换为规则三维数组。
            mask[tuple(all_coords.T)] = self.density_grid[0]
            #19.调整mask分辨率 mode='nearest'保持占有区域的离散属性
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[resolution] * 3, mode='nearest').squeeze(0).squeeze(0)
            #20.生成有效区域掩码 
            mask = (mask > density_thresh)
            #21.删除为训练区域
            sigmas = sigmas * mask

        #四、使用Marching Cubes提取网格
        #22.处理NaN 将NaN->0，避免 Marching Cubes 因为非法数值崩溃。
        sigmas = torch.nan_to_num(sigmas, 0)
        #23.转换到CPU和NumPy 
        #因为 mcubes.marching_cubes() 使用 NumPy 数组，所以要将 GPU Tensor 转换为 CPU NumPy 数
        sigmas = sigmas.cpu().numpy()
        #执行Marching Cubes vertices：网格顶点；triangles：三角形索引 范围[0,resolution-1]
        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
        #25.将坐标映射到[-1,1]  [0,resolution-1]=>[-1,1]
        vertices = vertices / (resolution - 1.0) * 2 - 1
        #26.转换数据类型
        vertices = vertices.astype(np.float32) 
        triangles = triangles.astype(np.int32) 

        #五、可见性裁剪
        ### visibility test.
        #27.判断不可见三角形 如果传入了数据集，就用训练相机判断三角形是否被看到
        if dataset is not None:
            visibility_mask = self.mark_unseen_triangles(vertices, triangles, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
            #28.删除不可见三角形 删除训练视角中完全不可见的三角形，dilation 用于扩大可见区域，避免边缘三角形被误删
            vertices, triangles = remove_masked_trigs(vertices, triangles, visibility_mask, dilation=self.opt.visibility_mask_dilation)
        #六、清理浮点网格
        ### reduce floaters by post-processing...
        #29.清理网格 删除小的孤立组件：删除浮点；合并过近的顶点；修复非流形边；清理无效面片
        #小于一定面数的孤立组件删除；直径过小的组件删除
        vertices, triangles = clean_mesh(vertices, triangles, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=True, remesh=False)
        
        #七、网格简化
        ### decimation
        #30.降低三角形数量 只有当当前三角形数量大于目标数量时，才进行简化 对应参数--decimate_target 300000
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)
        #八、保存内层网格
        #31.创建三角网格对象 process=False：禁止trimesh自动处理网格
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        #32.导出mesh_0.ply 作为初始网格
        mesh.export(os.path.join(save_path, f'mesh_0.ply'))

        #九、外层网格处理
        # for the outer mesh [1, inf]
        #说明场景范围更大，需要继续提取外层区域 [-2,2]^3,[-4,4]^3...
        if self.bound > 1:

            #33.设置外层网格分辨率
            reso = self.grid_size
            #控制外层环境网格的分辨率 --env_reso 256(默认)
            target_reso = self.opt.env_reso
            #代码将外层网格的简化目标减半，是经验设置,因为外层区域通常不需要像物体主体一样精细
            decimate_target //= 2 # empirical...

            all_indices = torch.arange(reso**3, device=device, dtype=torch.int)
            all_coords = raymarching.morton3D_invert(all_indices).cpu().numpy()

            # for each cas >= 1
            #34.遍历不同cascade
            for cas in range(1, self.cascade):
                #35.确定当前cascade的边界
                bound = min(2 ** cas, self.bound) #eg.cas=1 → bound=2  cas=2 → bound=4
                half_grid_size = bound / target_reso #half_grid_size 表示一个体素的大约半边长，用于后续裁剪边界

                # remap from density_grid
                #36.读取当前cascade的occupancy
                occ = torch.zeros([reso] * 3, dtype=torch.float32, device=device)
                #读取当前 cascade 的密度/占用网格 表示当前尺度下哪些位置可能存在场景
                occ[tuple(all_coords.T)] = self.density_grid[cas]

                # remove the center (before mcubes)
                # occ[reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4, reso // 4 : reso * 3 // 4] = 0

                # interpolate the occ grid to desired resolution to control mesh size...
                #37.插值到外层目标分辨率
                #使用三线性插值，将原来的 occupancy 网格调整到：[target_reso, target_reso, target_reso]
                occ = F.interpolate(occ.unsqueeze(0).unsqueeze(0), [target_reso] * 3, mode='trilinear').squeeze(0).squeeze(0)
                #转换为二值占用区域 将 occupancy 转换为布尔值：True False
                occ = torch.nan_to_num(occ, 0)
                occ = (occ > density_thresh).cpu().numpy()

                #39.提取外层表面 因为occ为二值数，用0.5作等值面阈值
                vertices_out, triangles_out = mcubes.marching_cubes(occ, 0.5)
                #40.映射外层坐标 先把坐标映射到：[-1,1] ，之后再根据当前 bound 放大
                vertices_out = vertices_out / (target_reso - 1.0) * 2 - 1 # range in [-1, 1]

                # remove the center (already covered by previous cascades)
                #41.删除与内层重叠的区域
                _r = 0.45
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {_r}) && (x >= -{_r}) && (y <= {_r}) && (y >= -{_r}) && (z <= {_r} ) && (z >= -{_r})')
                if vertices_out.shape[0] == 0: continue
                #42.缩放到当前cascade范围 将坐标[-1,1],缩放到当前的外层范围
                #eg.当前bound = 2,网格大致被映射到[-2,2]
                vertices_out = vertices_out * (bound - half_grid_size)

                # remove the out-of-AABB region
                #裁剪到训练AABB 读取训练场景的实际包围盒
                xmn, ymn, zmn, xmx, ymx, zmx = self.aabb_train.cpu().numpy().tolist()
                #然后扩大或缩小边界 防止网格顶点靠近外边界时产生不稳定面片
                xmn += half_grid_size
                ymn += half_grid_size
                zmn += half_grid_size
                xmx -= half_grid_size
                ymx -= half_grid_size
                zmx -= half_grid_size
                #44.删除越界顶点 删除超出训练 AABB 的顶点和相连三角形
                vertices_out, triangles_out = remove_selected_verts(vertices_out, triangles_out, f'(x <= {xmn}) || (x >= {xmx}) || (y <= {ymn}) || (y >= {ymx}) || (z <= {zmn} ) || (z >= {zmx})')

                # clean mesh
                #45.清理外层网络 清理外层网格中的小碎片 repair=False 因为外层网格可能是开放的，不一定需要进行完整的拓扑修复
                vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)

                if vertices_out.shape[0] == 0: continue

                # decimate
                #46.外层网格简化
                if decimate_target > 0 and triangles_out.shape[0] > decimate_target:
                    vertices_out, triangles_out = decimate_mesh(vertices_out, triangles_out, decimate_target, optimalplacement=False)

                vertices_out = vertices_out.astype(np.float32)
                triangles_out = triangles_out.astype(np.int32)
                #47.输出网格信息 顶点数量，面片数量
                print(f'[INFO] exporting outer mesh at cas {cas}, v = {vertices_out.shape}, f = {triangles_out.shape}')
                #48.对外层网格进行可见性裁剪 与内层网格相同，根据训练相机删除完全不可见的三角形
                if dataset is not None:
                    visibility_mask = self.mark_unseen_triangles(vertices_out, triangles_out, dataset.mvps, dataset.H, dataset.W).cpu().numpy()
                    vertices_out, triangles_out = remove_masked_trigs(vertices_out, triangles_out, visibility_mask, dilation=self.opt.visibility_mask_dilation)
                
                # vertices_out, triangles_out = clean_mesh(vertices_out, triangles_out, min_f=self.opt.clean_min_f, min_d=self.opt.clean_min_d, repair=False, remesh=False)
                #49.保存外层网格 mesh_1.ply...
                mesh_out = trimesh.Trimesh(vertices_out, triangles_out, process=False) # important, process=True leads to seg fault...
                mesh_out.export(os.path.join(save_path, f'mesh_{cas}.ply'))


    # phase 0 continuous training
    #	输入相机射线，沿射线采样三维点，调用 NeRFNetwork 预测密度和颜色，再通过体渲染合成最终像素颜色和深度。
    #1.函数输入，render不负责更新网络参数
    def render(self, rays_o, rays_d, index=None, dt_gamma=0, bg_color=None, perturb=False, max_steps=1024, T_thresh=1e-4, cam_near_far=None, shading='full', **kwargs):
        # rays_o 射线起点, rays_d: [N, 3] 射线方向表示输入有 N 条射线，每条射线由三维起点和三维方向表示
        # max_steps最大采样步数
        #T_thresh 射线透射率阈值，用于提前终止采样，cam_near_far相机指定的近、远裁剪距离
        # return: image: [N, 3], depth: [N] 返回image和depth
        
        #2.保存原始形状
        prefix = rays_o.shape[:-1]
        #3.展平射线
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] #射线数量
        device = rays_o.device #数据所在设备，通常为GPU

        # pre-calculate near far
        #4.计算射线与场景边界的交点
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        #5.使用相机提供的近远距离
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0]) #射线开始采样的位置
            fars = torch.minimum(fars, cam_near_far[:, 1]) #射线结束采样的位置
        # print((nears < fars).sum(), nears.shape[0])
        
        # mix background color 6.设置背景颜色
        if bg_color is None:
            bg_color = 1 #白色
            
        #7.读取 individual code 是每张训练图像对应的可学习隐变量，帮助网络表达：
        #不同图像间的颜色差异；光照差异；曝光变化；相机特性差异
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:#关闭了individual code，测试阶段使用第0个code
                ind_code = self.individual_codes[[0]]
        else:
            ind_code = None
        #8.创建字典
        results = {}
        
        #一、训练阶段
        if self.training:
            #一次性为所有射线生成采样点
            #9.训练阶段射线采样
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)
            #所有采样点对应三维坐标，每个采样点对应的射线方向，每个采样点在射线上的距离，记录每条射线包含哪些采样点
            #ts通常记录采样区间的起始和结束距离
            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            # ray-wise to point-wise
            #10.将射线级编码转换为点级编码，原来的 ind_code 通常是每条射线一个编码，但一个射线会对应多个采样点
            if ind_code is not None and ind_code.shape[0] > 1:
                #每个采样点一个code
                '''
                射线0 → 采样点0、1、2
                射线1 → 采样点3、4
                生成：
                code0、code0、code0、code1、code1
                '''               
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                ind_code = ind_code[flatten_rays]
            #11.归一化射线方向，这样可以保证方向只表示朝向，不受向量长度影响，这对视角相关颜色和高光预测很重要。
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            
            #12.调用NERFNetwork
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                #autocast 用于混合精度计算，可以减少显存占用并提高速度
                sigmas, rgbs, speculars = self(xyzs, dirs, ind_code, shading)
                #这里的self会调用NeRFNetwork.forward()，ind_code图像条件编码
                #密度sigma决定空间中是否存在表面，颜色决定该位置对当前视角的外观。

            #13.训练阶段提渲染合成
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh)
            #14.保存训练中间结果，用于后续损失计算
            results['num_points'] = xyzs.shape[0]
            results['xyzs'] = xyzs
            results['speculars'] = speculars
            results['weights'] = weights #每个采样点的权重
            results['weights_sum'] = weights_sum #每条射线的总权重
        
        #二、推理阶段，不需要保存梯度，重点是提高效率和减少显存使用。
        else:
            # allocate outputs 
            # output should always be float32! only network inference uses half.
            #15.初始化输出，即使网络内部使用 FP16，最终渲染结果仍然使用 FP32
            dtype = torch.float32
            #初始化
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            #16.初始化存活射线
            n_alive = N
            #表示当前仍然需要继续采样的射线。
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            #rays_t 表示每条射线当前采样到的位置，初始为 near
            rays_t = nears.clone() # [N]
            
            #17.推理阶段循环采样
            #推理阶段不是一次性采样所有点，而是循环进行,这样可以动态删除已经饱和的射线，节省计算。
            '''
            采样一批点
            → 网络预测
            → 累积颜色和深度
            → 删除已经结束的射线
            → 继续下一批
            '''            
            step = 0
            
            while step < max_steps:

                # count alive rays 18.统计剩余射线数量
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps 19.决定本轮采样步数
                # 这里根据剩余射线数量动态决定每条射线本轮采样几个点，最大不超过8
                n_step = max(min(N // n_alive, 8), 1)
                
                #20.CUDA射线采样，这一步沿当前仍然有效的射线进行采样
                #march_rays会在推理阶段分批生成采样点
                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, dt_gamma, max_steps)
                
                #21.推理阶段网络预测
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)#归一化方向
                #调用NeRFNetwork，预测密度，颜色，高光
                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    sigmas, rgbs, speculars = self(xyzs, dirs, ind_code, shading)
                #22.累积颜色和深度
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh)
                #composite_rays与composite_rays_train作用相同，但推理版本是原地更新，并且支持动态删除已经结束的射线。
                #23.删除已经结束的射线
                rays_alive = rays_alive[rays_alive >= 0]
                #如果：透射率很低；已经到达far；累积权重已经足够大

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')
                #更新采样步数
                step += n_step
        #三、添加背景颜色，weights_sum表示物体对该像素的覆盖程度
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        #四、恢复图像形状 [H,W,3] [N,3]
        image = image.view(*prefix, 3)

        # depth = torch.clamp(depth - nears, min=0) / (fars - nears) 恢复深度 [H,W]
        depth = depth.view(*prefix)
        #五、保存最终结果
        results['depth'] = depth
        results['image'] = image

        return results
    
    # phase 2 mesh光栅化渲染
    #先将当前三角网格光栅化到图像平面，再利用 NeRF 网络预测网格表面的颜色，最后得到渲染图像、深度和三角形索引
    #1.函数输入 MVP:	Model-View-Projection 模型视图投影矩阵 h0:原始图像高度 w0:原始图像宽度
    def render_stage1(self, rays_o, rays_d, mvp, h0, w0, index=None, bg_color=None, shading='full', **kwargs):
        #2.保存图像原始形状
        prefix = rays_d.shape[:-1]
        #3.展平观察方向 [H, W, 3] → [H×W, 3] 每个像素对应一个观察方向
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_d.shape[0] # N = B * N, in fact
        device = rays_d.device
            
        #一、超采样抗锯齿
        # do super-sampling
        #4.调整渲染分辨率 ssaa 表示 Super-Sampling Anti-Aliasing，超采样抗锯齿倍数
        if self.opt.ssaa > 1:
          #eg.h0 = 512,w0 = 512,ssaa = 2,则实际先渲染h=1024,w=1024再缩小到:512*512 可以减轻网格边缘锯齿
            h = int(h0 * self.opt.ssaa)
            w = int(w0 * self.opt.ssaa)
            # interpolate rays_d when ssaa > 1 ...
            #5.放大观察方向
            dirs = rays_d.view(h0, w0, 3) #先将观察方向恢复成图像格式 [N, 3] → [h0, w0, 3]
            #再放大到超采样分辨率 [h0, w0, 3] → [h, w, 3] 最后重新展平
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        #不实用超采样 直接使用原始分辨率
        else:
            h, w = h0, w0
            dirs = rays_d.contiguous()
        #归一化观察方向 观察方向会被输入 NeRF 的颜色网络，用于预测视角相关的颜色和高光
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

        #二、背景颜色和图像编码
        # mix background color
        #8.设置背景颜色
        if bg_color is None:
            bg_color = 1

        # [N, 3] to [h, w, 3]
        #9.调整背景颜色形状
        if torch.is_tensor(bg_color) and len(bg_color.shape) == 2:
            bg_color = bg_color.view(h0, w0, 3)
        #10.读取individual code
        #如果启用ind code训练时使用当前图像对应的编码；测试时使用第 0 个固定编码；没有启用时，设置为 None
        #可以表示不同图像之间的 光照变化、曝光变化、相机颜色差异
        if self.individual_dim > 0:
            if self.training:
                ind_code = self.individual_codes[index]
            # use a fixed ind code for the unknown test data.
            else:
                ind_code = self.individual_codes[[0]]
        else:
            ind_code = None

        results = {}

        #三、获取当前可优化网格
        #11.加载网格定顶点 初始网格顶点+可学习的顶点偏移量 
        #这意味着 Stage 1 可以通过反向传播调整网格顶点位置，使几何形状更加贴合真实图像
        vertices = self.vertices + self.vertices_offsets # [N, 3]
         
        #12.顶点投影到裁剪空间 首先给每个三维坐标增加其次坐标[x,y,z,1] 乘以MVP矩阵(包含模型变换、相机视图变换、投影变换)
        #得到裁剪空间坐标 [1,N_vertics,4]
        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4]
        
        #四、网格光栅化
        #13.使用nvdiffrast光栅化 将三角网格投影到图像平面
        #输入vertices_clip：投影后的顶点 self.triangles:三角形连接关系 (h, w)：渲染分辨率
        #输出rast，通常包含每个像素的重心坐标，深度，三角形索引
        #光栅化之后，每个图像像素都可以知道 是否落在哪个三角形内，哪个三角形，对应三角形哪个位置
        rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))

        #五、插值得到像素对应的三维坐标
        #14.插值三维顶点坐标 这里利用三角形的重心坐标，对三角形三个顶点进行插值
        #表示每个像素对应的三维表面坐标
        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles) # [1, H, W, 3]
        #15.插值得到物体掩码 给每个顶点设置值 1，然后插值得到像素掩码 [1, H, W, 1]
        #mask > 0：像素落在网格表面 mask = 0：像素没有被网格覆盖
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles) # [1, H, W, 1]
        #16.展平掩码和三维坐标  后面只对网格覆盖的像素计算颜色
        mask_flatten = (mask > 0).view(-1) #[H*W]
        xyzs = xyzs.view(-1, 3) #[H*W,3]

        #六、预测网格颜色
        #17.初始化RGB图像 创建全黑图像
        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        #18.只处理有效网格像素
        if mask_flatten.any():
          #19.调用NeRF颜色网络
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                mask_rgbs, masked_specular = self.rgb( # 调用 rgb() 根据网格表面点和观察方向预测颜色
                    
                    #20.enable_offset_nerf_grad 的作用
                    #xyzs[mask_flatten]xyzs[mask_flatten].detach()和的区别是是否允许梯度通过 NeRF 颜色网络传回网格顶点
                    #若切断梯度，这样只训练颜色网络，不通过颜色误差更新顶点偏移 颜色损失 → NeRF颜色网络
                    #开启时，颜色损失 → xyzs → 顶点坐标 → vertices_offsets 利用颜色误差帮助优化网格几何
                    xyzs[mask_flatten] if self.opt.enable_offset_nerf_grad 
                    else xyzs[mask_flatten].detach(), 
                    dirs[mask_flatten], ind_code, shading)
            #21.将颜色放回图像 只将网格覆盖区域的颜色写入图像 未被网格覆盖的像素仍然为[0,0,0]，后面用背景色填充
            rgbs[mask_flatten] = mask_rgbs.float()
        
        #七、抗锯齿处理
        #22.恢复图像形状
        rgbs = rgbs.view(1, h, w, 3)
        #23.分离alpha alphas 表示网格覆盖区域 物体区域：alpha接近1；背景区域：alpha接近0
        alphas = mask.float().detach() #detach() 表示先切断 mask 的梯度
        
        #24.对alpha进行可微抗锯齿 dr.antialias() 根据三角形边界进行抗锯齿处理 边缘像素不在只有0,1；可能是：0.2 0.5
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        #25对RGB进行抗锯齿 对颜色图像也进行边界抗锯齿，使网格轮廓更加平滑
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

        #八、计算最终图像和深度
        #26.计算物体颜色
        image = alphas * rgbs 
        #27.计算深度 rast[..., 2] 通常表示光栅化得到的深度值 
        # 这里的深度通常是投影空间或相机空间中的 z 值，不一定是射线方向上的欧氏距离
        depth = alphas * rast[0, :, :, [2]]
        #28.计算背景透射率 T = 1：完全是背景 T = 0：完全被网格覆盖
        T = 1 - alphas

        #九、记录三角形编号
        #29.获取每个像素对应的三角形id rast 的最后一个通道存储三角形编号
        # trig_id for updating trig errors 像素没被三角形覆盖 trig_id=-1 这个三角形编号后面用于统计三角形误差
        trig_id = rast[0, :, :, -1] - 1 # [h, w] 减去 1 是因为光栅化结果中的三角形编号通常从 1 开始，而 Python 数组索引从 0 开始

        #十、超采样结果缩小
        # ssaa
        #30.将结果缩小到原始分辨率 这一步可以降低锯齿，同时得到最终输出分辨率
        if self.opt.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            #31.缩小三角形编号图 三角形编号不能使用双线性插值，否则会生成不存在的编号，因此使用nearest 最近邻插值
            trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest').long()
        #32.保存三角形误差索引 这一步是 Stage 1 自适应细化的关键
        #训练时，程序根据：当前像素属于哪个三角形，把像素误差累积到对应三角形上：像素误差 → 三角形误差，
        #误差大的三角形后续会被细分，误差小的区域可能被简化
        self.triangles_errors_id = trig_id
        if getattr(self.opt, 'refine_multiround', False) or getattr(self.opt, 'refine_multibranch_eval', False):
            self.refinement_alpha = (1 - T).detach().reshape(*prefix)
        
        #十一、合成背景
        #33.加入背景颜色
        image = image + T * bg_color
        
        #十二、恢复输出形状 [H, W, 3]
        image = image.view(*prefix, 3) #[H, W, 3]
        depth = depth.view(*prefix) #[H,W]


        #十三、返回结果
        results['depth'] = depth
        results['image'] = image

        # tmp: visualize accumulated triangle error by abusing depth
        # error_val = self.triangles_errors[trig_id.view(-1)].view(*prefix)
        # error_cnt = self.triangles_errors_cnt[trig_id.view(-1)].view(*prefix)
        # cnt_mask = error_cnt > 0
        # error_val[cnt_mask] = error_val[cnt_mask] / error_cnt[cnt_mask].float()
        # results['depth'] = error_val

        return results

    @torch.no_grad()
    #	每个像素的渲染误差，累积到它所属的三角形上，为后续自适应细化提供依据
    def update_triangles_errors(self, loss): #loss[y,x]=当前像素的RGB误差
        # loss: [H, W], detached! 表示loss已经与计算图断开，不再进行梯度传播 这里的目的不是训练网络，而是统计三角形误差

        # always call after render_stage1, so self.triangles_errors_id is not None.
        #2.获取像素对应的三角形编号 triangles_errors_id在render_stage1()中生成
        #eg.triangles_errors_id =[[ 3,  3, -1],[ 5,  5,  8]] 展平后 indices = [3, 3, -1, 5, 5, 8]
        indices = self.triangles_errors_id.view(-1)
        #3.创建有效像素掩码
        mask = (indices >= 0) #mask = [True, True, False, True, True, True]
        
        #4.删除背景像素的三角形编号
        indices = indices[mask].contiguous() #indices = [3, 3, 5, 5, 8]
        #5.提取有效像素的误差 loss.view(-1)把误差图展平 [mask].contiguous()删除背景像素误差
        values = loss.view(-1)[mask].contiguous() #loss = [0.2, 0.5, 0.8, 0.1, 0.3, 0.7]
        # =>values = [0.2, 0.5, 0.1, 0.3, 0.7]

        #6.延迟加载torch_scatter 第一次执行时才导入 torch_scatter，之后重复使用已经加载的模块
        global TORCH_SCATTER

        if TORCH_SCATTER is None:
            import torch_scatter
            TORCH_SCATTER = torch_scatter

        #7.累加每个三角形总误差 scatter_add 根据 indices 将 values 累加到对应位置 eg.三角形3：累计误差 0.7
        TORCH_SCATTER.scatter_add(values, indices, out=self.triangles_errors)
        #8.统计每个三角形被观察次数，对每个有效像素累加 1 eg.三角形3被2个像素看到
        TORCH_SCATTER.scatter_add(torch.ones_like(values), indices, out=self.triangles_errors_cnt)
        #10.清空三角形索引 避免误用上一次渲染的三角形编号
        self.triangles_errors_id = None


    @torch.no_grad()
    # 根据相机可见范围标记未训练的网格区域，减少无效计算
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.density_bitfield.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)
        
        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):

            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.to(device), 0, 1)).float().unsqueeze(0) # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(self.glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1
            
            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1 # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = (mask == 0) # unseen faces by all cameras

        print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
        
        return mask # [N]


    @torch.no_grad()
    def mark_untrained_grid(self, dataset, S=64):
        
        # data: reference to the dataset object

        poses = dataset.poses # [B, 4, 4]
        intrinsics = dataset.intrinsics # [4] or [B/1, 4]
        cam_near_far = dataset.cam_near_far if hasattr(dataset, 'cam_near_far') else None # [B, 2]
  
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if isinstance(intrinsics, np.ndarray):
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = torch.chunk(intrinsics, 4, dim=-1)
        
        mask_cam = torch.zeros_like(self.density_grid)
        mask_aabb = torch.zeros_like(self.density_grid)

        # pc = []
        poses = poses.to(mask_cam.device)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # first, mark out-of-AABB region
                        mask_min = (cas_world_xyzs >= (self.aabb_train[:3] - half_grid_size)).sum(-1) == 3
                        mask_max = (cas_world_xyzs <= (self.aabb_train[3:] + half_grid_size)).sum(-1) == 3
                        mask_aabb[cas, indices] += (mask_min & mask_max).reshape(-1)

                        # second, mark out-of-camera region
                        # split pose to batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            cam_xyzs[:, :, 2] *= -1 # crucial, camera forward is negative now...

                            if torch.is_tensor(fx):
                                cx_div_fx = cx[head:tail] / fx[head:tail]
                                cy_div_fy = cy[head:tail] / fy[head:tail]
                            else:
                                cx_div_fx = cx / fx
                                cy_div_fy = cy / fy
                            
                            min_near = self.opt.min_near if cam_near_far is None else cam_near_far[head:tail, 0].unsqueeze(1)
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > min_near # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx_div_fx * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy_div_fy * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask = (mask_z & mask_x & mask_y).sum(0).bool().reshape(-1) # [N]

                            # for visualization
                            # pc.append(cas_world_xyzs[0][mask])

                            # update mask_cam 
                            mask_cam[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[((mask_cam == 0) | (mask_aabb == 0))] = -1

        print(f'[mark untrained grid] {((mask_cam == 0) | (mask_aabb == 0)).sum()} from {self.grid_size ** 3 * self.cascade}')

    
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if self.opt.stage > 0:
            return

        if not self.cuda_ray:
            return 
        
        ### update density grid

        with torch.no_grad():

            tmp_grid = - torch.ones_like(self.density_grid)
            
            # full update.
            if self.iter_density < 16:
                X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
                Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
                Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # query density
                                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                    sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                                # assign 
                                tmp_grid[cas, indices] = sigmas

            # partial update (half the computation)
            else:
                N = self.grid_size ** 3 // 4 # H * H * H / 4
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                    occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # query density
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                    # assign 
                    tmp_grid[cas, indices] = sigmas

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            
        if not self.opt.trainable_density_grid:
            self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        else:
            # update grid via a loss term.
            loss = F.mse_loss(self.density_grid[valid_mask], tmp_grid[valid_mask])

            # cascaded reg
            loss_density = 0
            if self.opt.lambda_density > 0:
                for cas in range(1, self.cascade):
                    loss_density = loss_density + (2 ** (cas - 1)) * self.opt.lambda_density * self.density_grid[cas][valid_mask[cas]].mean()

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        # density_thresh = 0 if self.iter_density < 64 else self.density_thresh
        self.density_bitfield = raymarching.packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')

        if not self.opt.trainable_density_grid:
            return None
        else:
            return loss + loss_density
