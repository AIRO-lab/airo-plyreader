# airo-plyreader

GPU 기반 포인트 클라우드 기둥 검출 파이프라인. `.ply` 파일을 읽어 색상 세그멘테이션, DBSCAN 클러스터링, PCA 분석으로 기둥 축과 박스를 추정하고, 결과를 주석이 입혀진 `.ply` 파일과 JSON으로 저장한다.

CUDA 12.x와 NVIDIA GPU가 필요하다 (`cupy-cuda12x`, `cuml-cu12`). ROI 및 HSV 선택은 Matplotlib/TkAgg를 사용하므로 `DISPLAY`가 있어야 한다.

## 사용법

```bash
uv sync
mkdir -p ply            # 여기에 .ply 파일을 넣는다
uv run python -m src.main      # 대화형 파이프라인 실행
uv run python -m src.viewer    # 이전 실행 결과 다시 보기
```

Docker:

```bash
docker compose --profile gpu up airo-plyreader-gpu
# 컨테이너 내부에서:
bash scripts/install_deps.sh && uv sync
```

결과는 `output/<plyname>-<timestamp>/` 아래에 생성된다.

파이프라인 파라미터(HSV 임계값, DBSCAN eps, PCA 임계값, 복셀 크기)는 `src/config.py`에 있다. 아키텍처 설명은 `CLAUDE.md` 참고.
