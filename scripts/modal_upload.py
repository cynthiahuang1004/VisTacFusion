"""Upload data to Modal volume in batches (one batch per object + weights)."""
import os
import modal

VOLUME_NAME = "vistacfusion-data"
SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
MESH_DIR = "/media/hdd2/ihsuan/gs_blender/meshes"

OBJECTS = [
    "pattern_01_2_lines_angle_1_2", "pattern_01_2_lines_angle_2",
    "pattern_01_2_lines_angle_3", "pattern_04_3_lines_angle_1",
    "pattern_04_3_lines_angle_2", "pattern_06_5_lines_angle_1",
    "pattern_31_rod", "pattern_32", "pattern_33",
    "pattern_35", "pattern_36", "pattern_37",
]
NEEDED_SUBDIRS = {"samples_g", "rgb", "raw_data", "norms", "calibration"}


def upload_sim_object(vol, obj):
    obj_path = os.path.join(SIM_ROOT, obj)
    if not os.path.isdir(obj_path):
        print(f"  SKIP {obj} (not found)")
        return

    with vol.batch_upload(force=True) as batch:
        for sess in sorted(os.listdir(obj_path)):
            sess_path = os.path.join(obj_path, sess)
            if not os.path.isdir(sess_path):
                continue
            sj = os.path.join(sess_path, "session.json")
            if os.path.exists(sj):
                batch.put_file(sj, f"/sim/{obj}/{sess}/session.json")
            for sensor in sorted(os.listdir(sess_path)):
                if not sensor.startswith("sensor_"):
                    continue
                sensor_path = os.path.join(sess_path, sensor)
                for sub in NEEDED_SUBDIRS:
                    sub_path = os.path.join(sensor_path, sub)
                    if os.path.isdir(sub_path):
                        batch.put_directory(sub_path,
                                            f"/sim/{obj}/{sess}/{sensor}/{sub}")
    print(f"  [done] {obj}")


def main():
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # --- Sim data (one batch per object) ---
    print("=== Uploading sim data (12 objects) ===")
    for i, obj in enumerate(OBJECTS):
        print(f"  [{i+1}/12] {obj} ...")
        upload_sim_object(vol, obj)

    # --- Real data ---
    print("=== Uploading real data ===")
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(REAL_ROOT, "/real")
    print("  [done] real data")

    # --- Meshes ---
    print("=== Uploading meshes ===")
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(MESH_DIR, "/meshes")
    print("  [done] meshes")

    # --- Encoder weights ---
    print("=== Uploading encoder weights ===")
    WEIGHTS = [
        ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/t3_large/encoder_mini.pth",
         "/weights/t3_large/encoder_mini.pth"),
        ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/t3_large/trunk.pth",
         "/weights/t3_large/trunk.pth"),
        ("/media/hdd2/ihsuan/VisTacFusion/weights/dinov3_vitl16_pretrain_lvd1689m.pth",
         "/weights/dinov3_vitl16.pth"),
        ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/mae_vitl16.pth",
         "/weights/mae_vitl16.pth"),
        ("/media/hdd2/ihsuan/VisTacFusion/pretrained_encoders/dav2_vitl14.pth",
         "/weights/dav2_vitl14.pth"),
        ("/media/hdd2/ihsuan/sparsh/weights/sparsh/dinov2_vitbase.ckpt",
         "/weights/sparsh_dinov2_base.ckpt"),
        ("/media/hdd2/ihsuan/sparsh/weights/sparsh/mae_vitbase.ckpt",
         "/weights/sparsh_mae_base.ckpt"),
        ("/media/hdd2/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth",
         "/weights/SITR_B18.pth"),
    ]
    with vol.batch_upload(force=True) as batch:
        for local, remote in WEIGHTS:
            if os.path.exists(local):
                mb = os.path.getsize(local) / 1e6
                print(f"  {os.path.basename(local)} ({mb:.0f} MB)")
                batch.put_file(local, remote)
            else:
                print(f"  MISSING: {local}")
    print("  [done] weights")

    # --- Rotation windows ---
    print("=== Uploading config files ===")
    rw = "/media/hdd2/ihsuan/VisTacFusion/ablation/simqty_filtered/real_rotation_windows.json"
    with vol.batch_upload(force=True) as batch:
        if os.path.exists(rw):
            batch.put_file(rw, "/configs/real_rotation_windows.json")
    print("  [done] configs")

    print("\n=== Upload complete ===")


if __name__ == "__main__":
    main()
