"""Compare the per-face-minimum zmap of build_real_filtered_new.get_zmap with an exact (barycentric) rasterization,
within the contact band (zmap - zmin < 1.6 mm). usage: python aligntac_newobj/check_zmap.py <mesh names...>"""
import sys, numpy as np, trimesh
sys.path.insert(0, "scripts"); import build_real_filtered_new as B

def exact_zmap(name):
    m = trimesh.load(f"{B.MESH_DIR}/{name}.obj", force="mesh"); V, F = m.vertices, m.faces
    LO = m.bounds[0][:2]; PPM = 20; RS = 82 * PPM; Z = np.full((RS, RS), 999.0)
    for f in F:
        v = V[f]; p = (v[:, :2] - LO) * PPM - 0.5            # pixel centres at (i + 0.5) / PPM
        x0, y0 = max(int(np.floor(p[:, 0].min())), 0), max(int(np.floor(p[:, 1].min())), 0)
        x1, y1 = min(int(np.ceil(p[:, 0].max())) + 1, RS), min(int(np.ceil(p[:, 1].max())) + 1, RS)
        if x1 <= x0 or y1 <= y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        a, b, c = p; d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12: continue                           # vertical face
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d; l3 = 1 - l1 - l2
        ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        z = l1 * v[0, 2] + l2 * v[1, 2] + l3 * v[2, 2]
        Z[ys[ins], xs[ins]] = np.minimum(Z[ys[ins], xs[ins]], z[ins])
    return Z.astype(np.float32)

if __name__ == "__main__":
    for n in sys.argv[1:]:
        old = B.get_zmap(n)[0]; new = exact_zmap(n); zmin = new.min()
        band = (new - zmin < 1.6) | (old - zmin < 1.6)
        diff = np.abs(np.minimum(old, zmin + 1.6) - np.minimum(new, zmin + 1.6))[band]
        print(f"{n:30s} contact-band px {band.sum():7d}  mean |diff| {diff.mean():.3f} mm  p95 {np.percentile(diff, 95):.3f}  max {diff.max():.3f}", flush=True)
        np.save(f"aligntac_newobj/zmap_exact_{n}.npy", new)
