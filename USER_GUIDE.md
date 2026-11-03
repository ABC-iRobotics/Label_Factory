# Label Factory – Docker felhasználói útmutató

Ez az útmutató a `Label_Factory_clean` Docker-alapú telepítését és a grafikus
felület alapvető használatát mutatja be. A rendszer RealSense RGB-D felvételből,
ismert STL modellekből metrikus 3D rekonstrukciót, objektumpózokat és képenkénti
annotációkat készít.

## 1. Előfeltételek

A gazdagépen szükséges:

- Linux;
- Docker Engine és a `docker compose` plugin;
- támogatott NVIDIA GPU és működő NVIDIA driver;
- NVIDIA Container Toolkit;
- Intel RealSense RGB-D kamera a kalibrációhoz és felvételhez;
- elegendő tárhely a Docker image, a képek és a rekonstrukció számára.

Ellenőrzés:

```bash
docker --version
docker compose version
nvidia-smi
```

A GPU Dockerből való elérésének ellenőrzése:

```bash
docker run --rm --gpus all \
  nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Ha ez nem működik, telepíteni és konfigurálni kell az NVIDIA Container
Toolkit csomagot, majd újra kell indítani a Dockert:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. Könyvtárak és modellek előkészítése

Lépj a repository gyökerébe:

```bash
cd ~/Label_Factory
```

Hozd létre a tartós könyvtárakat:

```bash
mkdir -p data calibration/cameras checkpoints/ism checkpoints/pem
```

A SAM-6D súlyfájlok nem kerülnek bele a Git repositoryba. Elvárt helyük:

```text
checkpoints/
├── ism/
│   ├── segment-anything/sam_vit_h_4b8939.pth
│   ├── dinov2/dinov2_vitl14_pretrain.pth
│   └── FastSAM/FastSAM-x.pt             # csak FastSAM használatakor
└── pem/
    ├── sam-6d-pem-base.pth
    └── mae_pretrain_vit_base.pth
```

## 3. GPU-profil kiválasztása

Az RTX 2000 Ada profil körülbelül 8 GiB VRAM-ra készült:

```text
pipeline.rtx2000.env
```

Az A100 profil nagyobb párhuzamosságot és host cache-t használ:

```text
pipeline.a100.env
```

Mindig ugyanazt a profilt használd a build és a futtatás során. A PointNet2
CUDA-kiegészítő a build során az adott GPU architektúrájára fordul.

## 4. Docker image építése

### RTX 2000 Ada

```bash
docker compose --env-file pipeline.rtx2000.env build label-factory
```

### NVIDIA A100

```bash
docker compose --env-file pipeline.a100.env build label-factory
```

Ha GPU-profilt, függőséget, Dockerfile-t vagy Python scriptet változtattál,
biztos újraépítéshez használd:

```bash
docker compose --env-file pipeline.rtx2000.env build --no-cache label-factory
```

Az első build hosszú lehet, mert két Conda környezetet, Blendert és a PointNet2
CUDA-kiegészítőt is elkészíti.

## 5. Kamera- és grafikusablak-hozzáférés

A Streamlit böngészős felület mellett a kalibráció, felvétel és néhány
vizualizáció natív ablakot nyit a Docker host kijelzőjén. Indítás előtt:

```bash
xhost +local:docker
```

A `compose.yaml` a teljes `/dev/bus/usb` könyvtárat átadja a containernek. A
RealSense kamera ellenőrzése a futó szolgáltatásban:

```bash
docker compose --env-file pipeline.rtx2000.env run --rm \
  --entrypoint /opt/conda/envs/label-factory/bin/python \
  label-factory -c \
  "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])"
```

Ezen a Docker/host konfiguráción a RealSense által használt USB ioctl műveletek
a bind mount és a device cgroup rule mellett is blokkolva maradhatnak. Erre
szolgál a külön `compose.capture.yaml` override. Csak helyi kalibrációhoz és
RGB-D felvételhez indítsd vele a szolgáltatást:

```bash
docker compose --env-file pipeline.rtx2000.env \
  -f compose.yaml -f compose.capture.yaml up label-factory
```

Ez az indítás `privileged` hozzáférést ad a containernek. Csak megbízható,
helyben épített Label Factory image-dzsel használd. A feldolgozáshoz és a távoli
szervercontainerhez nincs szükség erre az override-ra; azok továbbra is a
normál `compose.yaml` fájllal indíthatók.

Ha a lista üres:

- ellenőrizd az USB 3 kábelt és portot;
- zárd be a RealSense Viewert és minden más kamerát használó programot;
- indítsd újra a containert a kamera csatlakoztatása után;
- ellenőrizd a host USB-jogosultságait.

## 6. A felület indítása

### RTX 2000 Ada

```bash
docker compose --env-file pipeline.rtx2000.env up label-factory
```

### NVIDIA A100

```bash
docker compose --env-file pipeline.a100.env up label-factory
```

A felület ugyanazon a gépen:

```text
http://localhost:8501
```

Másik hálózati gépről:

```text
http://HOST_IP:8501
```

A host címei:

```bash
hostname -I
```

Háttérben történő indítás és logkövetés:

```bash
docker compose --env-file pipeline.rtx2000.env up -d label-factory
docker compose logs -f --tail=150 label-factory
```

## 7. Alapvető munkafolyamat

### Framework guide

Áttekintést ad a feldolgozás lépéseiről, bemeneteiről és kimeneteiről. Új
adatfelvétel előtt érdemes végigolvasni.

### Setup

1. Add meg vagy válaszd ki a dataset nevét. Szöveges név és számozott név is
   használható.
2. Töltsd fel az ismert objektumok STL modelljeit.
3. Ellenőrizd és mentsd az STL fájlokat.
   Opcionálisan tegyél textúrázott modellt (`<név>.obj`, `<név>.mtl` és a textúra
   képe) az `objects/textured/` mappába. Ha a modell teljes, a rendszer
   automatikusan használja a SAM-6D sablonok rendereléséhez, egyébként az STL-t.
   Az OBJ-nak ugyanabban a koordináta-rendszerben és mértékegységben kell lennie,
   mint az STL-nek. A beállítások között látható, melyik objektumhoz van textúra.
4. Ellenőrizd, hogy a szükséges környezetek és csomagok elérhetők-e.

A dataset a hoston a következő helyre kerül:

```text
data/<dataset-név>/
```

### Capture

1. Válassz csatlakoztatott RealSense kamerát.
2. Válassz meglévő kalibrációt, vagy készíts új ChArUco kalibrációt.
3. Állítsd be az RGB/depth felbontást és az FPS-t.
4. Rögzítsd a teljes valós jelenetet, lassan mozogva és nagy képi átfedéssel.

Javasolt körülbelül 250 képkocka. A felvétel ablakában a `Space` indítja vagy
szünetelteti a rögzítést; `Q` vagy `Esc` lezárja és menti azt.

Fő kimenetek:

```text
data/<dataset>/images/
data/<dataset>/depth_maps/
data/<dataset>/camera.json
```

### Reconstruction

A folyamat sorrendje:

1. CUDA SIFT feature extraction;
2. sequential feature matching;
3. sparse SfM és CPU bundle adjustment;
4. sparse pontfelhő szűrése;
5. CUDA PatchMatch MVS és stereo fusion;
6. dense pontfelhő szűrése;
7. RGB-D alapú metrikus Sim(3) skálázás.

Az RTX profil egy, az A100 profil négy PatchMatch workert használ. A
`Geometric consistency` kikapcsolása gyorsabb, de csökkentheti a dense
rekonstrukció megbízhatóságát. A `Replace existing reconstruction` kapcsoló
felülírhatja az előző rekonstrukciós eredményeket.

Fő kimenet:

```text
data/<dataset>/reconstruction/mvs_filtered_metric.ply
```

### Instance segmentation & Pose estimation

1. Ellenőrizd a bemeneteket.
2. Indítsd el a SAM-6D feldolgozást.
3. Vizsgáld meg az objektumspecifikus `vis_ism` és `vis_pem` eredményeket.

A SAM proposalok és DINO query feature-ök képkockánként közös cache-be
kerülnek. Az ISM és PEM modell több objektum feldolgozásakor csak egyszer
töltődik be. Az első futás létrehozza a cache-eket, ezért a későbbi futások
gyorsabbak lehetnek.

### Pose Fitting

A képkockánkénti pózjelöltekből pozícióklasztereket, majd klaszterenként
orientációkonszenzust készít. Több azonos típusú fizikai objektum külön
instance-ként jelenhet meg. Ha egy objektumhoz nincs elegendő stabil jelölt, a
felület figyelmeztetést mutat, a megtalált objektumok feldolgozása azonban
megmarad.

A „Min. silhouette IoU” mezővel kiszűrheted azokat a hasonló kinézetű objektumokat,
amelyek geometriája eltér a modellétől. A rendszer az illesztett CAD-modell
sziluettjét összeveti a pózbecslő által használt maszkokkal; a helyesen illesztett
objektumok jellemzően 0,8–0,95 közötti mediánt kapnak, az eltérő geometriájúak
észrevehetően alacsonyabbat. Az érték (0,7 körül érdemes kezdeni) csak akkor szűr, ha
nagyobb nullánál; az IoU-t a pózfájl minden példányhoz eltárolja, a küszöböt ezekből
a jó objektumokra kapott értékekből érdemes megválasztani.

### Scene & annotation

A metrikus rekonstrukcióból, kamera-pózokból, objektummodellekből és illesztett
pózokból képenkénti bounding box, instance mask, pózadat és vizuális overlay
készül.

### Review & publish

Vizsgáld meg az RGB-, depth-, mask-, overlay- és pózkimeneteket. Ezután állítsd
be a train/validation/test arányokat, a licencet és a Hugging Face repositoryt.
A publikálás pull requesten keresztül történik.

Hugging Face token átadása indításkor:

```bash
HF_TOKEN="hf_..." docker compose --env-file pipeline.rtx2000.env \
  up label-factory
```

A tokent ne tárold Gitben és ne írd be verziókövetett konfigurációs fájlba.

## 8. Adatok megőrzése és felülírás

A következő host könyvtárak bind mountként csatlakoznak:

```text
./data        -> /workspace/data
./calibration -> /workspace/calibration
./checkpoints -> SAM-6D checkpoint könyvtárak
```

A container törlése vagy újraépítése nem törli ezeket. A UI `Replace` vagy
`Overwrite` kapcsolói viszont felülírhatják az adott dataset korábbi
kimeneteit. Fontos adatoknál készíts másolatot a `data/<dataset>` könyvtárról.

## 9. Leállítás és frissítés

Leállítás:

```bash
docker compose down
xhost -local:docker
```

Új verzió letöltése és újraépítése:

```bash
git pull
docker compose --env-file pipeline.rtx2000.env build label-factory
docker compose --env-file pipeline.rtx2000.env up label-factory
```

## 10. Gyors ellenőrzések és hibakeresés

### A container nem látja a GPU-t

```bash
docker compose --env-file pipeline.rtx2000.env run --rm label-factory \
  nvidia-smi
```

Ha ez hibás, a host driver vagy az NVIDIA Container Toolkit nincs megfelelően
beállítva.

### A PyCOLMAP nem CUDA-képes

```bash
docker compose --env-file pipeline.rtx2000.env run --rm label-factory \
  python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
```

A dense PatchMatch futtatásához `True` szükséges; nincs CPU fallback.

### A SAM-6D nem látja a GPU-t

```bash
docker compose --env-file pipeline.rtx2000.env run --rm label-factory \
  /opt/conda/envs/sam6d/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### CUDA out of memory

RTX 2000 Ada esetén ellenőrizd, hogy valóban a `pipeline.rtx2000.env` profilt
használod. Zárd be a többi GPU-programot. MVS esetén csökkenthető a maximális
képméret vagy a source image-ek száma. A SAM-6D stage-eket ne indítsd több UI
sessionből párhuzamosan.

A100 40 GiB esetén az MVS workerek ideiglenesen kettőre csökkenthetők:

```bash
MVS_GPU_INDICES=0,0 docker compose --env-file pipeline.a100.env \
  up label-factory
```

### A UI nem érhető el

```bash
docker compose ps
docker compose logs --tail=150 label-factory
```

Ellenőrizd, hogy a `8501` port nincs-e más alkalmazás által használva.

### Natív ablak nem jelenik meg

```bash
echo "$DISPLAY"
xhost +local:docker
```

Távoli, headless szerveren az X11/Open3D ablakokhoz külön kijelzőtovábbítás
szükséges; maga a Streamlit UI ettől még böngészőből használható.

## 11. Hasznos Docker-parancsok

```bash
# Futó szolgáltatások
docker compose ps

# Utolsó 150 logsor
docker compose logs --tail=150 label-factory

# Folyamatos logkövetés
docker compose logs -f label-factory

# Shell indítása a containerben
docker compose --env-file pipeline.rtx2000.env run --rm \
  --entrypoint bash label-factory

# Container leállítása és eltávolítása; a bind mountolt adatok megmaradnak
docker compose down
```
