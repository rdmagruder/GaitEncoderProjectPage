# app.py
import io, uuid, tempfile, requests, json, torch, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'])

class FeatVector(BaseModel):
    features: list[float]  # length == 16

# ---- preload -------------------------------------------------
device = torch.device('cuda:0')
vae = torch.load('vae.pt', map_location=device).eval()
osim_path = 'model.osim'
joints   = ['time',
              'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
              'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
              'hip_flexion_ips', 'hip_adduction_ips', 'hip_rotation_ips',
              'knee_angle_ips', 'ankle_angle_ips','subtalar_angle_ips',
              'hip_flexion_contra', 'hip_adduction_contra', 'hip_rotation_contra',
              'knee_angle_contra', 'ankle_angle_contra', 'subtalar_angle_contra',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
              'arm_add_ips', 'arm_flex_ips', 'arm_rot_ips', 'elbow_flex_ips', 'pro_sup_ips',
              'arm_add_contra', 'arm_flex_contra', 'arm_rot_contra', 'elbow_flex_contra', 'pro_sup_contra']  # same order you use in numpy_to_storage

# --------------------------------------------------------------

def numpy_to_storage(labels, data, storage_file, datatype=None):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    # Old style
    if datatype is None:
        f = open(storage_file, 'w')
        f.write('name %s\n' % storage_file)
        f.write('datacolumns %d\n' % data.shape[1])
        f.write('datarows %d\n' % data.shape[0])
        f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
        f.write('endheader \n')
    # New style
    else:
        if datatype == 'IK':
            f.write('Coordinates\n')
        elif datatype == 'ID':
            f.write('Inverse Dynamics Generalized Forces\n')
        elif datatype == 'GRF':
            f.write('%s\n' % storage_file)
        elif datatype == 'muscle_forces':
            f.write('ModelForces\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' % data.shape[0])
        f.write('nColumns=%d\n' % data.shape[1])
        if datatype == 'IK':
            f.write('inDegrees=yes\n\n')
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write(
                "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
        elif datatype == 'ID':
            f.write('inDegrees=no\n')
        elif datatype == 'GRF':
            f.write('inDegrees=yes\n')
        elif datatype == 'muscle_forces':
            f.write('inDegrees=yes\n\n')
            f.write('This file contains the forces exerted on a model during a simulation.\n\n')
            f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write('Angles are in degrees.\n\n')

        f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()
@app.post('/reconstruct')
def reconstruct(v: FeatVector):
    z = torch.tensor(v.features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        x = vae.decoder(z)
        x = x * vae.normalize_std + vae.normalize_mean
    arr = x.cpu().numpy().reshape(-1, len(joints))

    tmp_mot = tempfile.NamedTemporaryFile(suffix='.mot', delete=False)
    numpy_to_storage(joints, arr, tmp_mot.name, datatype='IK')

    files = {
        'model_file': open(osim_path, 'rb'),
        'motion_file': open(tmp_mot.name, 'rb')
    }
    r = requests.post(
        'https://opensim-to-visualizer-api.onrender.com/convert-opensim-to-visualizer-json',
        files=files, timeout=90)
    r.raise_for_status()

    # host somewhere (S3 presigned, Cloudflare R2, etc.) or just return JSON
    json_url = upload_to_s3(r.content, key=f'{uuid.uuid4()}.json')
    return {'json_url': json_url}
