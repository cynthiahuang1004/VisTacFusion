"""Download Modal results with rate limit handling."""
import modal, os, time, re

vol = modal.Volume.from_name('vistacfusion-results')

time.sleep(2)
entries = list(vol.listdir('/'))
print(f"Found {len(entries)} runs in volume")

for entry in sorted(entries, key=lambda x: x.path):
    remote_name = entry.path.strip('/')
    m = re.match(r'ablation_(.+)_g3s_sim315$', remote_name)
    if m:
        local_name = f'ablation_g3s_sim315_{m.group(1)}'
    else:
        local_name = remote_name

    local_dir = os.path.join('outputs', local_name)
    os.makedirs(local_dir, exist_ok=True)

    time.sleep(2)
    try:
        files = list(vol.listdir(f'/{remote_name}'))
    except Exception as ex:
        print(f'  SKIP {remote_name} (rate limited: {ex})')
        continue

    if not files:
        print(f'  SKIP {remote_name} (empty)')
        continue

    for f in files:
        fname = os.path.basename(f.path)
        if fname.endswith(('.pt', '.json', '.yaml')):
            local_path = os.path.join(local_dir, fname)
            time.sleep(1)
            try:
                with open(local_path, 'wb') as fout:
                    for chunk in vol.read_file(f.path):
                        fout.write(chunk)
                print(f'  {local_name}/{fname}')
            except Exception as ex:
                print(f'  FAIL {local_name}/{fname}: {ex}')

print('Download complete.')
