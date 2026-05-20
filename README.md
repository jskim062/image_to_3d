# 🏗️ Hunyuan3D-2.1 Advanced Extension & BIM Pipeline

이 프로젝트는 Tencent의 **Hunyuan3D-2.1**을 확장하여, 다각도 실사 투영 복원, 듀얼 GPU 메모리 최적화, 그리고 생성된 3D 에셋을 건설정보모델링(BIM) 표준인 IFC 포맷으로 변환해 벽체 중공(Hollowing)을 지원하는 고급 엔지니어링 파이프라인입니다.

본 저장소는 다음의 세 가지 핵심 모듈로 구성되어 있습니다:
1. **🔄 4-to-6 Directional Multi-View Image Fusion System (`multiview_pipeline.py`)**
2. **⚡ Kaggle Dual-T4 GPU Optimization Notebooks (`kaggle/`)**
3. **🧱 GLB-to-IFC BIM Converter with Wall Hollowing (`glb_to_ifc.py`)**

---

## 📂 프로젝트 구조

```
c:/image_to_3d/
├── multiview_pipeline.py      # 멀티뷰 turnaround 복원 파이프라인 CLI 오케스트레이터
├── glb_to_ifc.py              # 솔리드 메쉬 -> 중공형 BIM IFC 포맷 변환 스크립트
├── test_multiview_system.py   # 로컬 CPU 환경용 통합 회귀 테스트 스위트
├── multiview_utils/           # 멀티뷰 코어 연산 모듈 (네임스페이스 충돌 방지 격리)
│   ├── image_utils.py         # 자동 시트 슬라이싱, 여백 센터링, 히스토그램 색상 정밀 정렬
│   └── multiview_paint_pipeline.py  # 6방향 실사 투영 텍스처 퓨전 및 seam-blending 서브클래스
├── kaggle/                    # Kaggle Dual T4 GPU 맞춤 최적화 실행 노트북
│   ├── hunyuan3d_2_1_dual_gpu.ipynb  # 단일 이미지 -> 3D (VRAM 극대화, 무적의 C++ JIT/OpenCV 복구)
│   └── hunyuan3d_2_1_multiview.ipynb # 턴어라운드 시트 -> 3D 멀티뷰 투영 전용 노트북 (GitHub 자동 연동)
└── colab/                     # Colab용 보조 테스트 노트북
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

## 🧱 3. GLB-to-IFC BIM 변환 파이프라인 (Hollowing 벽체 중공 특화)
생성된 3D 메쉬 오브젝트(.glb)를 표준 BIM 엔지니어링 규격인 IFC4 포맷으로 자동 번역하고, 내부를 비워 실제 벽체 두께를 연출하는 고성능 토목/건축용 파이프라인입니다.

### 핵심 기술 요약
- **버텍스 노멀 오프셋 쉘링 ($v_{\text{new}} = v - thickness \times \vec{n}$)**: 메쉬의 표면 노멀(법선 벡터)의 내부 방향으로 면을 축소 복제하고 법선을 뒤집어(Winding Inversion) **완벽한 이중 벽체 구조**를 만들어냅니다.
- **자동 단위 스케일 조정**: Blender나 AI 생성 모델의 무작위 단위를 실제 BIM 뷰어에서 정합적으로 작동하는 **국제표준 미터(Meter) 규격**으로 고밀도 선형 오조정합니다.
- **BIM 개체 정식 클래스 바인딩**: 단순히 파일만 변환하는 것이 아니라, IFC 계통도 내부의 `IfcWall`, `IfcSlab`, `IfcColumn` 등 유효한 BIM 엔티티로 완벽하게 등록합니다.

### 💻 사용 방법 (CLI 터미널)
```bash
# 기본 변환 (솔리드 메쉬 -> IFC 미터 스케일 정합화)
python glb_to_ifc.py my_model.glb output_bim.ifc --class IfcWall

# 벽체 두께 15cm 중공(Hollowing) 벽체 변환 기법 적용
python glb_to_ifc.py my_model.glb output_wall.ifc --hollow --thickness 0.15 --class IfcWall
```

---

## 🧪 4. 로컬 자가 진단 및 회귀 검증
인프라 이송이나 깃허브 코드 변경 시, CPU만 탑재된 로컬 개발 머신에서도 PyMeshLab, Pybind11 및 CUDA 그래픽 드라이버 장치를 완전히 가상 모방(Mocking)하여 파이프라인 전체 로직의 정합성을 검증할 수 있는 진단 회귀 테스트가 탑재되어 있습니다.

```bash
# 로컬 개발 환경에서 즉시 6개 코어 기능 정합성 테스트 가동
python test_multiview_system.py
```
*(성공 시 CPU 환경 상에서도 0.3초 내에 모든 턴어라운드 및 파이프라인 분배 연산 로직이 정상적임을 'OK' 사인으로 판정합니다.)*
