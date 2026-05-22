# 🏗️ Hunyuan3D-2.1 Advanced Extension & BIM Pipeline

이 프로젝트는 Tencent의 **Hunyuan3D-2.1**을 확장하여, 다각도 실사 투영 복원, 듀얼 GPU 메모리 최적화, 그리고 생성된 3D 에셋을 라이노(Rhino) 및 디지털 가공(3D 프린터/CNC)용 초정밀 CAD 포맷으로 변환하는 하이엔드 엔지니어링 파이프라인입니다.

본 저장소는 다음의 네 가지 핵심 모듈로 구성되어 있습니다:
1. **🔄 4-to-6 Directional Multi-View Image Fusion System (`multiview_pipeline.py`)**
2. **⚡ Kaggle Dual-T4 GPU Optimization Notebooks (`kaggle/`)**
3. **📐 CAD & Digital Fabrication Mesh Exporter (`glb_to_cad.py`)** [추천]
4. **🧱 GLB-to-IFC BIM Converter with Wall Hollowing (`glb_to_ifc.py`)**

---

## 📂 프로젝트 구조

c:/image_to_3d/
├── multiview_pipeline.py         # 멀티뷰 turnaround 복원 파이프라인 CLI 오케스트레이터
├── architectural_cad_pipeline.py # 건축 전용 초경량 CAD 및 제작용 3D 복원 파이프라인 [NEW]
├── glb_to_cad.py                 # CAD 및 디지털 패브리케이션용 메쉬 가속 Exporter
├── glb_to_ifc.py                 # 솔리드 메쉬 -> 중공형 BIM IFC 포맷 변환 스크립트
├── test_multiview_system.py      # 로컬 CPU 환경용 통합 회귀 테스트 스위트
├── test_architectural_cad.py     # 건축 CAD 기하학적 연산 특화 유닛 테스트 스위트 [NEW]
├── multiview_utils/              # 멀티뷰 코어 연산 모듈 (네임스페이스 충돌 방지 격리)
│   ├── image_utils.py            # 자동 시트 슬라이싱, 여백 센터링, 히스토그램 색상 정밀 정렬
│   └── multiview_paint_pipeline.py  # 6방향 실사 투영 텍스처 퓨전 및 seam-blending 서브클래스
├── kaggle/                       # Kaggle Dual T4 GPU 맞춤 최적화 실행 노트북
│   ├── hunyuan3d_2_1_dual_gpu.ipynb  # 단일 이미지 -> 3D (VRAM 극대화, 무적의 C++ JIT/OpenCV 복구)
│   ├── hunyuan3d_2_1_multiview.ipynb # 턴어라운드 시트 -> 3D 멀티뷰 투영 전용 노트북 (GitHub 자동 연동)
│   └── hunyuan3d_2_1_cad_fabrication.ipynb # 건축 전용 경량화 CAD/디지털 패브리케이션 특화 노트북 [NEW]
└── colab/                        # Colab용 보조 테스트 노트북
```

---

## 🔄 1. 4방향/6방향 멀티뷰 실사 투영 복원 시스템
기존의 단일 이미지 복원 방식은 보이지 않는 뒷면이나 측면을 AI의 상상력에만 의존하여 부정확하게 생성합니다. 이 모듈은 **앞면, 왼면, 뒷면, 오른면, (윗면, 아랫면)**의 실제 사진이 포함된 가로형 turnaround 캐릭터 시트나 스프라이트를 잘라내어 메쉬에 정교하게 직접 투영합니다.

### 핵심 기술 요약
- **시트 자동 슬라이싱 및 비율 감지**: 이미지 가로세로 비율이 $5.0$ 이상이면 **6방향 시트**, 미만이면 **4방향 시트**로 지능형 자동 판별 및 동일 간격 정밀 분할을 수행합니다.
- **10% 여백 정렬 패딩**: 슬라이싱된 오브젝트의 배경을 제거한 후 정중앙에 정렬하고 $10\%$ 여백을 추가하여, 텍스처 래핑 경계면의 신축 현상을 방지합니다.
- **히스토그램 기반 일광 정렬 (`align_multiview_colors`)**: 측면이나 뒷면 이미지의 조명/색감을 앞면 기준으로 정렬하여, 투영 경계면의 색상 이음새(Seam)를 수학적으로 완화합니다.
- **카메라 좌표 정밀 매핑**: Hunyuan3D 고유의 카메라 매핑 순서(`0: Front, 1: Right, 2: Back, 3: Left, 4: Top, 5: Bottom`)에 맞춰 잘려진 이미지를 알맞게 재배열하여 좌우 뒤바뀜 버그를 철저히 예방합니다.

### 💻 사용 방법 (로컬 / Kaggle 터미널)
```bash
# 기본 실행 (4방향 또는 6방향 이미지 자동 감지 및 3D 모델 복원)
python multiview_pipeline.py --sheet ./my_sheets/character_turnaround.png --view_order front_left_back_right --output_dir ./output_model

# 특정 메쉬 재사용 (메쉬 기하구조 모델링 단계 건너뛰고 텍스처 투영/블렌딩만 고속 재연산)
python multiview_pipeline.py --sheet ./my_sheets/character_turnaround.png --mesh ./output_model/base_geometry.glb --resolution 512
```

---

## ⚡ 2. Kaggle Dual-T4 GPU 최적화 (OOM 프리미엄 노트북)
Kaggle의 듀얼 T4 GPU 세션 환경에서 VRAM 파편화와 Out of Memory(OOM), JIT 컴파일러 크래시를 차단하고 100% 신뢰성 있는 빌드를 지원하는 엔지니어링 기법을 적용했습니다.

### 핵심 최적화 설계
1. **서브프로세스 격리 기술**: 용량이 매우 큰 텍스처링 모듈을 서브프로세스로 분리하여 실행합니다. 연산 종료 시 **운영체제가 VRAM을 100% 강제 해제**하므로 메모리 누수가 없습니다.
2. **GPU 분산 배치 (`Accelerate Dispatch`)**: 형상 생성(Shape) 단계에서 초대형 DiT 모델의 레이어 가중치를 GPU 0과 GPU 1에 균형 있게 실시간으로 분산 로드합니다.
3. **삼중 레이어 JIT 빌드 시스템**: C++ 모듈(`mesh_inpaint_processor`) 빌드 실패를 원천 봉쇄하기 위해 `setup.py` 빌드, PyTorch JIT 컴파일, 쉘 스크립트 빌드의 3단계 백업 구조를 적용하고, 최종 실패 시에도 **OpenCV Telea 인페인팅 알고리즘**으로 자동 전환되어 절대 중단되지 않습니다.
4. **노이즈 로그 정제**: Headless 환경에서 무수히 쏟아지는 Hugging Face 다운로드 게이지와 `tqdm` 진행 바를 싱글 라인으로 억제하여 콘솔 시인성을 크게 개선했습니다.

### 💻 사용 방법 (Kaggle)
1. Kaggle Notebook에서 `Import notebook`을 선택하여 `kaggle/hunyuan3d_2_1_multiview.ipynb`를 불러옵니다.
2. 우측 세션 옵션에서 **Accelerator**: `GPU T4 x2` 및 **Internet**: `On`을 켭니다.
3. 데이터셋 추가(Add Data) 기능을 통해 턴어라운드 시트 이미지를 등록합니다.
4. 위에서부터 순서대로 셀을 가동하면 실사 3D 모델이 `/kaggle/working/output_multiview/` 디렉토리에 복원됩니다.

---

## 📐 3. CAD & Digital Fabrication Mesh Exporter (`glb_to_cad.py`)
이 모듈은 AI가 생성한 유기적이고 복잡한 비선형(비정형) 3D 모델을 실제 설계 소프트웨어(라이노, 카티아 등) 및 공장 가공(3D 프린팅, CNC)에 바로 적용할 수 있도록 초정밀 엔지니어링 포맷으로 복구·최적화·변환해 줍니다.

> [!TIP]
> **왜 IFC 변환 대신 이 방식을 쓰나요?**
> 곡선이나 유체적인 흐름을 가진 비정형 건축물은 다각형 형태의 IFC로 직접 넘기면 수정 불가능한 무거운 돌덩어리가 됩니다. 대신, AI 모델의 삼각형 메쉬를 사각형 정렬(Quad-Mesh) 상태로 변환한 후 **Rhino SubD**를 이용해 **완벽한 수학적 곡선(NURBS) 솔리드**로 가공하여 수출하는 것이 세계 표준 워크플로우입니다.

### 핵심 기능 및 기술
- **메쉬 자가 복구 & 청소**: 인접 정점 병합(`merge_vertices`), 법선 벡터 재정렬(`fix_normals`), 불완전 면 제거를 통해 구멍이 없는 완벽한 Watertight 메쉬로 자동 복원합니다.
- **곡률 대비 적응형 메시 감축 (Curvature-Adaptive Decimation)**: 평평한 평면 부위는 삼각형 개수를 대폭 간소화하고, 부드러운 곡선 부위의 디테일은 촘촘하게 유지하여 가벼우면서도 완벽한 형태를 유지합니다.
- **용도별 맞춤 가속 포맷 추출**:
  1.  `[이름]_quad_ready.obj`: 라이노의 **QuadRemesh** 엔진이 가장 깔끔하게 인식할 수 있도록 고르게 분포된 메쉬 구조를 제공합니다.
  2.  `[이름]_fabrication.stl`: 3D 프린터 및 CNC 가공용 표준 포맷으로, 실제 미터법 규격(m, cm, mm) 단위로 정밀하게 스케일이 사전 계산되어 출력됩니다.
  3.  `[이름]_pointcloud.ply`: 고밀도 컬러 텍스처를 손실 없이 보존한 포인트 클라우드 포맷으로, 레빗(Revit) 등에 스캔 참조 데이터로 즉시 가볍게 로딩할 수 있습니다.

### 💻 사용 방법 (CLI 터미널)
```bash
# 기본 CAD 변환 (자동 스케일 감지, 기본 mm 단위로 실사 3D 모델에서 3개 포맷 동시 추출)
python glb_to_cad.py my_model.glb ./output_cad/

# 정밀 가공 최적화: 삼각형 면 개수를 50%로 줄이고, 실제 cm 단위 크기로 스케일하여 내보내기
python glb_to_cad.py my_model.glb ./output_cad/ --decimate 0.5 --unit cm
```

### 🏛️ 비정형/유기적 건축 라이노(Rhino) 연동 프로 워크플로우
1. `glb_to_cad.py`를 통해 생성된 `[이름]_quad_ready.obj`를 라이노로 가져옵니다 (`Import`).
2. 라이노 명령창에 `QuadRemesh`를 입력하고 실행하여 삼각형 메쉬를 정밀한 **사각 메쉬(Quad Mesh)**로 바꿉니다.
3. 사각 메쉬를 선택하고 `ToSubD` 명령어를 입력하여 면이 부드럽게 이어지는 **SubD(Subdivision Surface)** 모델로 바꿉니다.
4. 마지막으로 SubD 모델을 선택하고 `Convert` 명령어를 통해 **NURBS(비정형 서피스 솔리드)**로 최종 전환합니다.
5. 완성된 NURBS 솔리드는 마우스 클릭으로 벽체 두께를 조절하거나, 완벽한 수학적 곡선 상태로 **STEP/IGES**로 내보내 기계 가공에 적용할 수 있습니다.

---

## 🧪 4. 로컬 자가 진단 및 회귀 검증
이 프로젝트는 인프라 이송이나 깃허브 코드 변경 시, CPU만 탑재된 로컬 개발 머신에서도 PyMeshLab, Pybind11 및 CUDA 그래픽 드라이버 장치를 완전히 가상 모방(Mocking)하여 파이프라인 전체 로직의 정합성을 검증할 수 있는 진단 회귀 테스트가 탑재되어 있습니다.

```bash
# 로컬 개발 환경에서 멀티뷰 텍스처 파이프라인 정합성 테스트 가동
python test_multiview_system.py

# 로컬 개발 환경에서 신규 건축 CAD 기하학적 연산 특화 테스트 가동
python test_architectural_cad.py
```
*(성공 시 CPU 환경 상에서도 0.3초 내에 모든 기하 연산 및 슬라이싱 로직이 정상적임을 'OK' 사인으로 판정합니다.)*

---

## 🏗️ 5. 건축 전용 초경량 CAD & 디지털 제작 파이프라인 (`architectural_cad_pipeline.py`)

기존의 색상 텍스처 합성을 생략하고, 단 한 장의 2D 스케치/투시도로부터 실제 디지털 패브리케이션 장비 가공(3D 프린팅, CNC 밀링, 와플 그리드 커팅)에 즉시 투입 가능한 정밀 기하 데이터(STL, PLY, quad-friendly OBJ, DXF/SVG 단면 슬라이스)를 생성하는 경량형 초고속 파이프라인입니다.

### 핵심 기술 요약
- **초경량 1-Stage 가속**: 무거운 텍스처(Paint) 모델을 배제하고 형상(Shape) 단계만 탑재하여 **VRAM을 10GB 미만으로 절약**하고, 20초 만에 3D 모델을 복원합니다.
- **지면 고정 장치 (Ground Locking)**: 3D 복원 특유의 둥글고 불안정한 바닥면 부위를 설정한 비율(기본 5%)만큼 칼로 자르듯 평평하게 다듬고, 바닥을 수치 기준 좌표 원점($Z=0$)으로 완벽하게 밀착 정렬시킵니다.
- **기하학적 표면 노이즈 제거**: 마칭 큐브 알고리즘 고유의 복셀화 노이즈를 `Laplacian` 및 부피 수축 방지용 `Taubin` 스무딩 필터로 해결하여 Rhino NURBS 변환 및 가공에 최적화된 매끄러운 곡선 표면을 추출합니다.
- **수평 등고 단면 슬라이싱 (DXF/SVG Contours)**: 사용자가 지정한 등간격(예: 0.2m)마다 3D 메쉬를 수평 절단하여 모든 레이어의 위치가 3D로 완벽히 정렬된 3D DXF 파일과 각 레이저 커팅을 위한 개별 2D SVG 파일 대량 추출을 지원합니다.

### 💻 사용 방법 (로컬 / Kaggle 터미널)
```bash
# 단 한 장의 입면 컨셉 이미지로부터 지면 고정, Laplacian 스무딩 및 0.2 간격 슬라이싱 적용 실행
python architectural_cad_pipeline.py --image ./building_concept.png --lock_ground --ground_ratio 0.05 --smoothing_method laplacian --slicing_interval 0.2 --output_dir ./output_arch

# 여러 방향의 턴어라운드 건물 도면 시트를 슬라이싱하여 정면 기준으로 3D 형상 복원 가동
python architectural_cad_pipeline.py --sheet ./building_turnaround.png --num_views 4 --lock_ground --slicing_interval 0.1 --output_dir ./output_arch
```
*(결과물이 저장되는 디렉토리에 정밀 스케일링된 Watertight STL, Rhino SubD용 OBJ, 고밀도 PLY 포인트 클라우드, 레이저 가공을 위한 DXF/SVG 단면 도면들이 정밀 패키징되어 저장됩니다.)*
