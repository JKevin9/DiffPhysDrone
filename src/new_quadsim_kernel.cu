#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

namespace
{

    template <typename scalar_t>
    __global__ void render_cuda_kernel(
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> canvas,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> flow,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> balls,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders_h,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> voxels,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> R,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> R_old,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> pos,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> pos_old,
        float drone_radius,
        int n_drones_per_group,
        float azimuth_min,
        float azimuth_max,
        float elevation_min,
        float elevation_max)
    {

        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        const int B = canvas.size(0);
        const int H = canvas.size(1);
        const int W = canvas.size(2);
        if (c >= B * H * W)
            return;
        const int b = c / (H * W);
        const int u = (c % (H * W)) / W;
        const int v = c % W;

        // LiDAR光线方向计算
        const scalar_t elevation = elevation_min + (elevation_max - elevation_min) *
                                                       static_cast<scalar_t>(u) / (H - 1);
        const scalar_t azimuth = azimuth_min + (azimuth_max - azimuth_min) *
                                                   static_cast<scalar_t>(v) / (W - 1);

        // 在LiDAR坐标系中的方向向量 (前:x, 右:y, 下:z)
        const scalar_t elevation_rad = elevation * M_PI / 180.0;
        const scalar_t azimuth_rad = azimuth * M_PI / 180.0;
        const scalar_t dir_x = cos(elevation_rad) * cos(azimuth_rad);
        const scalar_t dir_y = cos(elevation_rad) * sin(azimuth_rad);
        const scalar_t dir_z = sin(elevation_rad);

        // 使用旋转矩阵转换到世界坐标系
        scalar_t dx = R[b][0][0] * dir_x + R[b][0][1] * dir_y + R[b][0][2] * dir_z;
        scalar_t dy = R[b][1][0] * dir_x + R[b][1][1] * dir_y + R[b][1][2] * dir_z;
        scalar_t dz = R[b][2][0] * dir_x + R[b][2][1] * dir_y + R[b][2][2] * dir_z;

        const scalar_t ox = pos[b][0];
        const scalar_t oy = pos[b][1];
        const scalar_t oz = pos[b][2];

        scalar_t min_dist = 15;
        // scalar_t t = (-1 - oz) / dz;
        // if (t > 0)
        //     min_dist = t;

        // others
        const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
        // for (int i = batch_base; i < batch_base + n_drones_per_group; i++)
        // {
        //     if (i == b || i >= B)
        //         continue;
        //     scalar_t cx = pos[i][0];
        //     scalar_t cy = pos[i][1];
        //     scalar_t cz = pos[i][2];
        //     scalar_t r = 0.15;
        //     // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        //     scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        //     scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        //     scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        //     scalar_t d = b * b - 4 * a * c;
        //     if (d >= 0)
        //     {
        //         r = (-b - sqrt(d)) / (2 * a);
        //         if (r > 1e-5)
        //         {
        //             min_dist = min(min_dist, r);
        //         }
        //         else
        //         {
        //             r = (-b + sqrt(d)) / (2 * a);
        //             if (r > 1e-5)
        //                 min_dist = min(min_dist, r);
        //         }
        //     }
        // }

        // balls
        for (int i = 0; i < balls.size(1); i++)
        {
            scalar_t cx = balls[batch_base][i][0];
            scalar_t cy = balls[batch_base][i][1];
            scalar_t cz = balls[batch_base][i][2];
            scalar_t r = balls[batch_base][i][3];
            scalar_t a = dx * dx + dy * dy + dz * dz;
            scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
            scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
            scalar_t d = b * b - 4 * a * c;
            if (d >= 0)
            {
                r = (-b - sqrt(d)) / (2 * a);
                if (r > 1e-5)
                {
                    min_dist = min(min_dist, r);
                }
                else
                {
                    r = (-b + sqrt(d)) / (2 * a);
                    if (r > 1e-5)
                        min_dist = min(min_dist, r);
                }
            }
        }

        // cylinders
        for (int i = 0; i < cylinders.size(1); i++)
        {
            scalar_t cx = cylinders[batch_base][i][0];
            scalar_t cy = cylinders[batch_base][i][1];
            scalar_t r = cylinders[batch_base][i][2];
            scalar_t a = dx * dx + dy * dy;
            scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
            scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
            scalar_t d = b * b - 4 * a * c;
            if (d >= 0)
            {
                r = (-b - sqrt(d)) / (2 * a);
                if (r > 1e-5)
                {
                    min_dist = min(min_dist, r);
                }
                else
                {
                    r = (-b + sqrt(d)) / (2 * a);
                    if (r > 1e-5)
                        min_dist = min(min_dist, r);
                }
            }
        }
        for (int i = 0; i < cylinders_h.size(1); i++)
        {
            scalar_t cx = cylinders_h[batch_base][i][0];
            scalar_t cz = cylinders_h[batch_base][i][1];
            scalar_t r = cylinders_h[batch_base][i][2];
            scalar_t a = dx * dx + dz * dz;
            scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
            scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
            scalar_t d = b * b - 4 * a * c;
            if (d >= 0)
            {
                r = (-b - sqrt(d)) / (2 * a);
                if (r > 1e-5)
                {
                    min_dist = min(min_dist, r);
                }
                else
                {
                    r = (-b + sqrt(d)) / (2 * a);
                    if (r > 1e-5)
                        min_dist = min(min_dist, r);
                }
            }
        }

        // balls
        for (int i = 0; i < voxels.size(1); i++)
        {
            scalar_t cx = voxels[batch_base][i][0];
            scalar_t cy = voxels[batch_base][i][1];
            scalar_t cz = voxels[batch_base][i][2];
            scalar_t rx = voxels[batch_base][i][3];
            scalar_t ry = voxels[batch_base][i][4];
            scalar_t rz = voxels[batch_base][i][5];
            scalar_t tx1 = (cx - rx - ox) / dx;
            scalar_t tx2 = (cx + rx - ox) / dx;
            scalar_t tx_min = min(tx1, tx2);
            scalar_t tx_max = max(tx1, tx2);
            scalar_t ty1 = (cy - ry - oy) / dy;
            scalar_t ty2 = (cy + ry - oy) / dy;
            scalar_t ty_min = min(ty1, ty2);
            scalar_t ty_max = max(ty1, ty2);
            scalar_t tz1 = (cz - rz - oz) / dz;
            scalar_t tz2 = (cz + rz - oz) / dz;
            scalar_t tz_min = min(tz1, tz2);
            scalar_t tz_max = max(tz1, tz2);
            scalar_t t_min = max(max(tx_min, ty_min), tz_min);
            scalar_t t_max = min(min(tx_max, ty_max), tz_max);
            if (t_min < min_dist && t_min < t_max && t_min > 0)
                min_dist = t_min;
        }

        canvas[b][u][v] = min_dist;
    }

    template <typename scalar_t>
    __global__ void nearest_pt_cuda_kernel(
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> nearest_pt,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> balls,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders_h,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> voxels,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> pos,
        float drone_radius,
        int n_drones_per_group)
    {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int B = nearest_pt.size(1);
        const int j = idx / B;
        if (j >= nearest_pt.size(0))
            return;
        const int b = idx % B;
        // assert(j < pos.size(0));
        // assert(b < pos.size(1));

        const scalar_t ox = pos[j][b][0];
        const scalar_t oy = pos[j][b][1];
        const scalar_t oz = pos[j][b][2];

        scalar_t min_dist = max(1e-3f, oz + 1);
        scalar_t nearest_ptx = ox;
        scalar_t nearest_pty = oy;
        scalar_t nearest_ptz = min(-1., oz - 1e-3f);

        // others
        const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
        // for (int i = batch_base; i < batch_base + n_drones_per_group; i++)
        // {
        //     if (i == b || i >= B)
        //         continue;
        //     scalar_t cx = pos[j][i][0];
        //     scalar_t cy = pos[j][i][1];
        //     scalar_t cz = pos[j][i][2];
        //     scalar_t r = 0.15;
        //     scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        //     dist = max(1e-3f, sqrt(dist) - r);
        //     if (dist < min_dist)
        //     {
        //         min_dist = dist;
        //         nearest_ptx = ox + dist * (cx - ox);
        //         nearest_pty = oy + dist * (cy - oy);
        //         nearest_ptz = oz + dist * (cz - oz);
        //     }
        // }

        // balls
        for (int i = 0; i < balls.size(1); i++)
        {
            scalar_t cx = balls[batch_base][i][0];
            scalar_t cy = balls[batch_base][i][1];
            scalar_t cz = balls[batch_base][i][2];
            scalar_t r = balls[batch_base][i][3];
            scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
            dist = max(1e-3f, sqrt(dist) - r);
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_ptx = ox + dist * (cx - ox);
                nearest_pty = oy + dist * (cy - oy);
                nearest_ptz = oz + dist * (cz - oz);
            }
        }

        // cylinders
        for (int i = 0; i < cylinders.size(1); i++)
        {
            scalar_t cx = cylinders[batch_base][i][0];
            scalar_t cy = cylinders[batch_base][i][1];
            scalar_t r = cylinders[batch_base][i][2];
            scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
            dist = max(1e-3f, sqrt(dist) - r);
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_ptx = ox + dist * (cx - ox);
                nearest_pty = oy + dist * (cy - oy);
                nearest_ptz = oz;
            }
        }
        for (int i = 0; i < cylinders_h.size(1); i++)
        {
            scalar_t cx = cylinders_h[batch_base][i][0];
            scalar_t cz = cylinders_h[batch_base][i][1];
            scalar_t r = cylinders_h[batch_base][i][2];
            scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
            dist = max(1e-3f, sqrt(dist) - r);
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_ptx = ox + dist * (cx - ox);
                nearest_pty = oy;
                nearest_ptz = oz + dist * (cz - oz);
            }
        }

        // voxels
        for (int i = 0; i < voxels.size(1); i++)
        {
            scalar_t cx = voxels[batch_base][i][0];
            scalar_t cy = voxels[batch_base][i][1];
            scalar_t cz = voxels[batch_base][i][2];
            scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
            scalar_t rx = min(max_r, voxels[batch_base][i][3]);
            scalar_t ry = min(max_r, voxels[batch_base][i][4]);
            scalar_t rz = min(max_r, voxels[batch_base][i][5]);
            scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
            scalar_t pty = cy + max(-ry, min(ry, oy - cy));
            scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
            scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
            dist = sqrt(dist);
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_ptx = ptx;
                nearest_pty = pty;
                nearest_ptz = ptz;
            }
        }
        nearest_pt[j][b][0] = nearest_ptx;
        nearest_pt[j][b][1] = nearest_pty;
        nearest_pt[j][b][2] = nearest_ptz;
    }

    template <typename scalar_t>
    __global__ void rerender_backward_cuda_kernel(
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> depth,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dddp,
        float azimuth_span,
        float elevation_span,
        int H,
        int W)
    {

        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        const int B = dddp.size(0);
        const int h = dddp.size(2);
        const int w = dddp.size(3);
        if (c >= B * h * w)
            return;
        const int b = c / (h * w);
        const int u = (c % (h * w)) / w;
        const int v = c % w;

        // 计算角度间隔
        const scalar_t d_azimuth = azimuth_span / (W - 1) * M_PI / 180.0;     // 转换为弧度
        const scalar_t d_elevation = elevation_span / (H - 1) * M_PI / 180.0; // 转换为弧度

        // 获取中心深度值
        const scalar_t d_center = depth[b][0][u * 2][v * 2];

        // 计算水平和垂直方向的深度梯度
        scalar_t d_az = 0, d_el = 0;
        if (v > 0 && v < W - 1)
        {
            d_az = (depth[b][0][u * 2][v * 2 + 1] - depth[b][0][u * 2][v * 2 - 1]) / (2 * d_azimuth);
        }
        if (u > 0 && u < H - 1)
        {
            d_el = (depth[b][0][(u + 1) * 2][v * 2] - depth[b][0][(u - 1) * 2][v * 2]) / (2 * d_elevation);
        }

        // 计算法向量 (归一化)
        const scalar_t norm = sqrt(1 + d_az * d_az + d_el * d_el);
        if (norm > 1e-5)
        {
            dddp[b][0][u][v] = -1.0 / norm;
            dddp[b][1][u][v] = d_az / norm;
            dddp[b][2][u][v] = d_el / norm;
        }
        else
        {
            dddp[b][0][u][v] = -1.0;
            dddp[b][1][u][v] = 0.0;
            dddp[b][2][u][v] = 0.0;
        }
    }

    // ... (nearest_pt_cuda_kernel 保持不变) ...

} // namespace

void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float azimuth_min,
    float azimuth_max,
    float elevation_min,
    float elevation_max)
{
    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&]
                                                              { render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                                                    canvas.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    flow.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                    balls.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    cylinders.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    cylinders_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    voxels.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    R.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    R_old.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                    pos.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                    pos_old.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                    drone_radius,
                                                                    n_drones_per_group,
                                                                    azimuth_min,
                                                                    azimuth_max,
                                                                    elevation_min,
                                                                    elevation_max); }));
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float azimuth_min,
    float azimuth_max,
    float elevation_min,
    float elevation_max)
{
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    const int H = depth.size(2) / 2; // 原始高度
    const int W = depth.size(3) / 2; // 原始宽度
    const float azimuth_span = azimuth_max - azimuth_min;
    const float elevation_span = elevation_max - elevation_min;

    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&]
                                                                        { rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                                                              depth.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                              dddp.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                              azimuth_span,
                                                                              elevation_span,
                                                                              H,
                                                                              W); }));
}

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group)
{
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&]
                                                               { nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                                                     nearest_pt.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     balls.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     cylinders.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     cylinders_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     voxels.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     pos.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                     drone_radius,
                                                                     n_drones_per_group); }));
}
