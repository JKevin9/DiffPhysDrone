// Procedural geometry only. The original dynamics kernels are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__device__ float clampf(float x, float a, float b) { return fminf(b, fmaxf(a,x)); }
__device__ void local(const float* obj, const float* p, float* q, bool point) {
    for(int j=0;j<3;j++) {
        q[j]=0;
        for(int k=0;k<3;k++) q[j]+=obj[7+k*3+j]*(p[k]-(point?obj[1+k]:0));
    }
}
__device__ float intersect(const float* obj, const float* p, const float* d, float best) {
    float q[3],v[3]; local(obj,p,q,true); local(obj,d,v,false);
    int type=int(obj[0]); const float* r=obj+4;
    if(type==1) {
        float lo=-INFINITY,hi=INFINITY;
        for(int k=0;k<3;k++) {
            if(fabsf(v[k])<1e-9f) { if(fabsf(q[k])>r[k]) return best; }
            else { float a=(-r[k]-q[k])/v[k],b=(r[k]-q[k])/v[k]; lo=fmaxf(lo,fminf(a,b)); hi=fminf(hi,fmaxf(a,b)); }
        }
        if(hi>=fmaxf(lo,0.0f)) { float t=lo>1e-5f?lo:hi; if(t>1e-5f) best=fminf(best,t); }
    } else if(type==2) {
        float a=0,b=0,c=-1;
        for(int k=0;k<3;k++){float s=1/(r[k]*r[k]); a+=v[k]*v[k]*s;b+=q[k]*v[k]*s;c+=q[k]*q[k]*s;}
        float disc=b*b-a*c;
        if(disc>=0 && a>1e-12f){float t=(-b-sqrtf(disc))/a;if(t<=1e-5f)t=(-b+sqrtf(disc))/a;if(t>1e-5f)best=fminf(best,t);}
    } else if(type==3) {
        float a=v[0]*v[0]+v[1]*v[1], b=q[0]*v[0]+q[1]*v[1], c=q[0]*q[0]+q[1]*q[1]-r[0]*r[0];
        float disc=b*b-a*c;
        if(disc>=0 && a>1e-12f) {
            float root=sqrtf(disc);
            for(int s=-1;s<=1;s+=2) {float t=(-b+s*root)/a; if(t>1e-5f && fabsf(q[2]+t*v[2])<=r[2])best=fminf(best,t);}
        }
        if(fabsf(v[2])>1e-9f) for(int s=-1;s<=1;s+=2) {float t=(s*r[2]-q[2])/v[2];float x=q[0]+t*v[0],y=q[1]+t*v[1];if(t>1e-5f && x*x+y*y<=r[0]*r[0])best=fminf(best,t);}
    }
    return best;
}
__global__ void rays_kernel(const float* objects,int count,const float* pos,const float* dirs,float* out,int rays,int batch,float maxdist) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=batch*rays)return;
    int b=i/rays;const float* p=pos+b*3;const float* d=dirs+i*3;
    float best=maxdist;bool hit=false;float dd=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
    for(int n=0;n<count;n++) {
        const float* obj=objects+(b*count+n)*17;if(obj[0]==0)continue;
        float dot=0;for(int k=0;k<3;k++)dot+=(obj[1+k]-p[k])*d[k];
        float t=clampf(dot/dd,0,best), ds=0;
        for(int k=0;k<3;k++){float x=p[k]+t*d[k]-obj[1+k];ds+=x*x;}
        if(ds>obj[16]*obj[16])continue;
        float next=intersect(obj,p,d,best);if(next<best){hit=true;best=next;}
    }
    out[i]=hit?best:INFINITY;
}
__device__ void nearest(const float* obj,const float* p,float* result) {
    float q[3],s[3];local(obj,p,q,true);const float* r=obj+4;int type=int(obj[0]);
    if(type==1){for(int k=0;k<3;k++)s[k]=clampf(q[k],-r[k],r[k]);}
    else if(type==3){float rad=sqrtf(q[0]*q[0]+q[1]*q[1]);float scale=fminf(1.0f,r[0]/fmaxf(rad,1e-10f));s[0]=q[0]*scale;s[1]=q[1]*scale;s[2]=clampf(q[2],-r[2],r[2]);}
    else {
        float norm=0;for(int k=0;k<3;k++)norm+=q[k]*q[k]/(r[k]*r[k]);
        if(norm<=1){for(int k=0;k<3;k++)s[k]=q[k];}
        else {
            float lo=0,hi=obj[16]*(sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2])+obj[16]);
            for(int iter=0;iter<28;iter++){float mid=(lo+hi)/2,val=0;for(int k=0;k<3;k++){float z=r[k]*q[k]/(mid+r[k]*r[k]);val+=z*z;}if(val>1)lo=mid;else hi=mid;}
            for(int k=0;k<3;k++)s[k]=r[k]*r[k]*q[k]/(hi+r[k]*r[k]);
        }
    }
    for(int k=0;k<3;k++){result[k]=obj[1+k];for(int j=0;j<3;j++)result[k]+=obj[7+k*3+j]*s[j];}
}
__global__ void nearest_kernel(const float* objects,int count,const float* points,float* out,int queries,int batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=batch*queries)return;
    int b=i/queries;const float* p=points+3*i;float best=INFINITY,result[3]={0,0,0};
    for(int n=0;n<count;n++) {
        const float* obj=objects+(b*count+n)*17;if(obj[0]==0)continue;
        float ds=0;for(int k=0;k<3;k++){float x=p[k]-obj[1+k];ds+=x*x;}
        if(sqrtf(ds)-obj[16]>best)continue;
        float q[3];nearest(obj,p,q);float dist=0;for(int k=0;k<3;k++){float x=q[k]-p[k];dist+=x*x;}
        dist=sqrtf(dist);if(dist<best){best=dist;for(int k=0;k<3;k++)result[k]=q[k];}
    }
    for(int k=0;k<3;k++)out[3*i+k]=result[k];
}
void validate(torch::Tensor t) {TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==torch::kFloat32,"Expected contiguous CUDA float32 tensor");}
torch::Tensor raycast(torch::Tensor objects,torch::Tensor pos,torch::Tensor dirs,double maxdist) {
    validate(objects);validate(pos);validate(dirs);
    TORCH_CHECK(objects.dim()==3 && objects.size(2)==17 && dirs.dim()==3 && dirs.size(2)==3 && pos.dim()==2 && pos.size(1)==3,"Invalid geometry shapes");
    TORCH_CHECK(objects.size(0)==dirs.size(0) && pos.size(0)==dirs.size(0) && objects.device()==pos.device() && pos.device()==dirs.device() && maxdist>0,"Geometry batch/device mismatch");
    c10::cuda::CUDAGuard guard(objects.device());
    auto out=torch::empty({dirs.size(0),dirs.size(1)},dirs.options());int total=out.numel();
    rays_kernel<<<(total+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(objects.data_ptr<float>(),objects.size(1),pos.data_ptr<float>(),dirs.data_ptr<float>(),out.data_ptr<float>(),dirs.size(1),dirs.size(0),maxdist);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor closest(torch::Tensor objects,torch::Tensor points) {
    validate(objects);validate(points);
    TORCH_CHECK(objects.dim()==3 && objects.size(2)==17 && points.dim()==3 && points.size(2)==3 && objects.size(0)==points.size(0) && objects.device()==points.device(),"Invalid closest-point shapes/devices");
    c10::cuda::CUDAGuard guard(objects.device());auto out=torch::empty_like(points);int total=points.numel()/3;
    nearest_kernel<<<(total+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(objects.data_ptr<float>(),objects.size(1),points.data_ptr<float>(),out.data_ptr<float>(),points.size(1),points.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("raycast",&raycast);m.def("closest",&closest);}
