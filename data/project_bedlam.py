import os, csv, glob, numpy as np, torch, cv2, smplx

ROOT = os.environ.get("BEDLAM_ROOT", "/workspace/shapectrl/data/BEDLAM")
MODELS = os.environ.get("SMPLX_MODELS", "/workspace/shapectrl/data/models")
GGT = f"{ROOT}/gendered_ground_truth"
SENSOR_W, SENSOR_H = 36.0, 20.25
IMG_W, IMG_H = 1280, 720

def set_orientation(rotate_flag):
    global SENSOR_W, SENSOR_H, IMG_W, IMG_H
    if rotate_flag:
        SENSOR_W, SENSOR_H, IMG_W, IMG_H = 20.25, 36.0, 720, 1280
    else:
        SENSOR_W, SENSOR_H, IMG_W, IMG_H = 36.0, 20.25, 1280, 720

_models = {
    'male':   smplx.create(MODELS,'smplx',gender='male',  ext='npz',num_betas=10,flat_hand_mean=True,use_pca=False),
    'female': smplx.create(MODELS,'smplx',gender='female',ext='npz',num_betas=10,flat_hand_mean=True,use_pca=False),
}

def smplx_verts(poses, betas, trans, gender):
    m=_models[str(gender)]
    t=lambda a: torch.tensor(a).unsqueeze(0).float()
    o=m(betas=t(betas),global_orient=t(poses[:3]),body_pose=t(poses[3:66]),
        left_hand_pose=t(poses[75:120]),right_hand_pose=t(poses[120:165]),
        jaw_pose=t(poses[66:69]),leye_pose=t(poses[69:72]),reye_pose=t(poses[72:75]),
        transl=t(trans))
    return o.vertices[0], o.joints[0]

def unreal2cv2(p):
    p=np.roll(p,2,1); return p*np.array([1.0,-1.0,1.0])

def focal_mm2px(fl,sens,fp): return (fl/sens)*fp*2
def cam_int(fl):
    return np.array([[focal_mm2px(fl,SENSOR_W,IMG_W/2.),0,IMG_W/2.],
                     [0,focal_mm2px(fl,SENSOR_H,IMG_H/2.),IMG_H/2.],[0,0,1]])

def cam_rotmat(body_yaw,pitch,yaw,roll):
    ry,_=cv2.Rodrigues(np.array([[0,((yaw)/180)*np.pi,0]],float))
    rp,_=cv2.Rodrigues(np.array([pitch/180*np.pi,0,0]).reshape(3,1))
    rr,_=cv2.Rodrigues(np.array([0,0,roll/180*np.pi]).reshape(3,1))
    return np.matmul(rr,np.matmul(rp,ry))

def get_cam_trans(body_trans,cam_t):
    ct=unreal2cv2(np.reshape(np.array(cam_t)/100,(1,3)))
    bt=unreal2cv2(np.reshape(np.array(body_trans)/100,(1,3)))
    return bt-ct

def get_j0(beta,gender):
    _,jl=smplx_verts(np.zeros(165),beta,torch.zeros(3),gender)
    return jl[0].detach().cpu().numpy()

def global_orient(pose,beta,transl,gender,body_yaw,cam_pitch,cam_yaw,cam_roll,j0=None):
    br,_=cv2.Rodrigues(np.array([[0,((body_yaw-90+cam_yaw)/180)*np.pi,0]],float))
    rp,_=cv2.Rodrigues(np.array([cam_pitch/180*np.pi,0,0]).reshape(3,1))
    rr,_=cv2.Rodrigues(np.array([0.,0,cam_roll/180*np.pi]).reshape(3,1))
    final=np.matmul(rr,rp)
    tc=np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
    tbr=np.matmul(br,tc)
    w_go=cv2.Rodrigues(np.dot(tbr,cv2.Rodrigues(pose[:3])[0]))[0].T[0]
    if j0 is None: j0=get_j0(beta,gender)
    l_t=np.matmul(tbr,j0.T).T+np.matmul(tbr,transl.T).T
    c_go=cv2.Rodrigues(np.dot(final,cv2.Rodrigues(w_go)[0]))[0].T[0]
    c_tr=np.matmul(final,l_t.T).T-j0
    return c_go,c_tr,final

def project(points,cam_trans,K):
    pts=points+cam_trans
    K=torch.tensor(K).float()
    pp=pts/pts[:,-1].unsqueeze(-1)
    return torch.einsum('ij,kj->ki',K,pp.float()).detach().cpu().numpy()

def parse_seq(be_seq_path, seq_name):
    rows=list(csv.reader(open(be_seq_path)))
    bodies=[]; in_seq=False
    for r in rows:
        if len(r)<10: continue
        if r[1]=='Group':
            in_seq = (f"sequence_name={seq_name};" in r[9])
        elif r[1]=='Body' and in_seq:
            name=r[2]; sub='_'.join(name.split('_')[:-1]); mid=name.split('_')[-1]
            sf=1
            for kv in r[9].split(';'):
                if kv.startswith('start_frame='): sf=int(kv.split('=')[1])
            bodies.append((sub,mid,float(r[3]),float(r[4]),float(r[5]),float(r[6]),sf))
    return bodies

def load_cam(cam_csv):
    rows=list(csv.DictReader(open(cam_csv)))
    return rows

def project_frame(seq_name, img_ind, bodies, cam_rows, rotate_flag=False, return_cam=False):
    c=cam_rows[img_ind]
    cam_t=[float(c['x']),float(c['y']),float(c['z'])]
    cam_pitch=-float(c['pitch']); cam_yaw=-float(c['yaw'])
    cam_roll=-float(c['roll'])+(90 if rotate_flag else 0)
    fl=float(c['focal_length']); K=cam_int(fl)
    crot=cam_rotmat(0,cam_pitch,cam_yaw,cam_roll)
    out=[]; params=[]
    for (sub,mid,X,Y,Z,yaw,sf) in bodies:
        npz=f"{GGT}/{sub}/moving_body_para/{mid}/motion_seq.npz"
        if not os.path.exists(npz): continue
        d=np.load(npz,allow_pickle=True)
        idx=img_ind+sf
        if idx>=d['poses'].shape[0]: continue
        pose=d['poses'][idx].astype(np.float64); transl=d['trans'][idx].astype(np.float64)
        beta=d['betas'][:10].astype(np.float64); gender=str(d['gender'])
        ct=get_cam_trans([X,Y,Z],cam_t)
        ct=np.matmul(crot,ct.T).T
        c_go,c_tr,_=global_orient(pose,beta,transl,gender,yaw,cam_pitch,cam_yaw,cam_roll)
        pose_cam=pose.copy(); pose_cam[:3]=c_go
        full_trans=(c_tr+ct).reshape(3)
        if return_cam:
            params.append(dict(body=f"{sub}_{mid}", pose_cam=pose_cam.astype(np.float32),
                               betas=beta.astype(np.float32), trans_cam=full_trans.astype(np.float32),
                               gender=gender, K=K.astype(np.float32)))
        else:
            v3d,_=smplx_verts(pose_cam,beta,c_tr,gender)
            v2d=project(v3d,torch.tensor(ct),K)
            out.append(v2d[:,:2])
    return params if return_cam else out

if __name__=="__main__":
    import sys
    SEQDIR=sys.argv[2] if len(sys.argv)>2 else "20221010_3_1000_batch01hand"
    seq=sys.argv[3] if len(sys.argv)>3 else "seq_000000"
    img_ind=int(sys.argv[1]) if len(sys.argv)>1 else 0
    rotate_flag='closeup' in SEQDIR; set_orientation(rotate_flag)
    base=f"{ROOT}/{SEQDIR}/ground_truth/{SEQDIR}/ground_truth"
    be=f"{ROOT}/{SEQDIR}/ground_truth/{SEQDIR}/be_seq.csv"
    bodies=parse_seq(be,seq)
    print("bodies:",[(b[0],b[1],b[6]) for b in bodies])
    cam_rows=load_cam(f"{base}/camera/{seq}_camera.csv")
    SP="/tmp/claude-0/-workspace/3fcff7b3-fb83-4fe8-88c1-8ab68224e2f3/scratchpad"
    os.system(f'tar xf {ROOT}/{SEQDIR}/mp4/{SEQDIR}_mp4.tar -C {SP} {SEQDIR}/mp4/{seq}.mp4 2>/dev/null')
    mp4=f"{SP}/{SEQDIR}/mp4/{seq}.mp4"
    os.system(f'ffmpeg -v error -y -i {mp4} -vf "select=eq(n\\,{img_ind})" -vframes 1 {SP}/bedlam_f.png')
    img=cv2.imread(f"{SP}/bedlam_f.png")
    cols=[(0,255,0),(0,200,255),(255,100,0)]
    for i,v2d in enumerate(project_frame(seq,img_ind,bodies,cam_rows,rotate_flag=rotate_flag)):
        for x,y in v2d[::8].astype(int):
            if 0<=x<IMG_W and 0<=y<IMG_H: cv2.circle(img,(x,y),2,cols[i%3],-1)
    cv2.imwrite(f"/workspace/bedlam_proj_{seq}_{img_ind}.jpg",img)
    print(f"saved /workspace/bedlam_proj_{seq}_{img_ind}.jpg")
