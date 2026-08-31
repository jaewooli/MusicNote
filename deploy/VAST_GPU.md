# Vast.ai GPU MT3

목적: 평소에는 현재 CPU 서버만 돌리고, 정밀 채보가 필요할 때만 GPU를 쓴다.
현재 CPU는 실시간의 약 13배다 — 13초 클립이 약 3분, 앙상블(기본)이면 약 6분.

두 가지 방식이 있고, **온디맨드가 기본이다.**

| | 온디맨드 (`remote`) | 서버리스 (`vast`) |
|---|---|---|
| 기동 | 손으로 start/stop | 오토스케일러가 배정 |
| 콜드스타트 | 최초 생성 1회만. 이후 start는 1분 안쪽 | 유휴 후 매 세션마다 (실측 약 12분) |
| 유휴 비용 | 스토리지만 ($0.13~0.40 /GB/월) | 0 |
| GPU 비용 | 켜져 있는 동안만 ($0.05~0.09/시간) | 요청 처리 동안만 |
| 이미지 | `--target runtime` | `--target serverless` (PyWorker 포함) |

GPU 임대료는 사실상 공짜다 (20시간 = 약 $1). 실제 비용은 **콜드스타트 시간**과
**정지 중 스토리지 임대료**이고, 서버리스는 전자를 매 세션마다 문다. 예측 불가능한
실사용자 트래픽이 생기기 전까지는 온디맨드가 맞다.

## 구조

```text
현재 CPU 서버 (musicnote)
  └ mt3_bridge.py
       ├ BACKEND=local   ──▶ 같은 호스트 pm2 워커 (파일 경로 전달)
       ├ BACKEND=remote  ──▶ 빌린 GPU 박스의 :8732 (FLAC 업로드)   ← 기본 권장
       └ BACKEND=vast    ──▶ vastai SDK ─▶ 오토스케일러 ─▶ PyWorker ─▶ :8732
```

핵심: **`backend/mt3_worker.py`를 그대로 모델 서버로 쓴다.** 로컬 pm2 워커와 같은
파일이며, 원격에서는 `MT3_DEVICE=cuda`와 base64 오디오 입력만 다르다.
`remote`는 PyWorker를 거치지 않고 모델 서버에 직접 HTTP로 붙는다.

업로드는 원본이 아니라 **16 kHz mono FLAC**이다. MT3가 어차피 16 kHz로 리샘플하므로
원본을 보내면 대부분이 낭비다. 실측: 13초 클립 2.43 MB → 0.22 MB.

## 이미지

`deploy/vast/Dockerfile`은 스테이지 빌드다. 나중 레이어에서 파일을 지워도 그
파일을 만든 레이어에는 그대로 남으므로, 정리는 `trim` 스테이지에서 하고 살아남은
트리만 최종 스테이지로 복사한다.

```bash
# 온디맨드용 (기본)
deploy/vast/build.sh docker.io/<user>/musicnote-mt3:2 --push

# 서버리스용
MT3_TARGET=serverless deploy/vast/build.sh docker.io/<user>/musicnote-mt3:2-sl --push
```

빌드 호스트가 arm64면 `qemu-user-static` + `binfmt-support`가 필요하다. vast 워커는
x86_64이고, `--platform`을 빼면 docker가 조용히 호스트 아키텍처로 빌드해서 워커에서
실행이 안 된다.

체크포인트는 로컬 워커의 디렉터리에서 스테이징한다 (기본 `~/mt3-ckpts`). CPU
기준선을 만든 바로 그 가중치를 굽기 위해서다. 기본은 `yourmt3`만 굽는다;
`MT3_BAKE_MODELS`로 바꿀 수 있다.

## 배포 절차 (온디맨드)

### 1. 키

```bash
pip install vastai
# ~/.config/vastai/vast_api_key 에 두고 퍼미션 600. 프런트엔드·저장소 금지.
```

### 2. 인스턴스 생성

전부 `deploy/vast/gpu.sh`에 들어 있다. 손으로 할 이유가 없다.

```bash
deploy/vast/gpu.sh up       # 임대 → /health 대기 → 앱 전환   (2~4분)
deploy/vast/gpu.sh status   # 무엇이 떠 있고 앱이 어디를 보는지
deploy/vast/gpu.sh down     # 파기 → 앱을 로컬 CPU로 되돌림
```

아래는 그 스크립트가 실제로 하는 일이고, **`--ssh`를 쓰면 안 된다.**

```bash
vastai create instance <offer_id> --image <레지스트리>/musicnote-mt3:3 \
  --disk 20 --env '-p 8732:8732' --onstart-cmd 'bash' \
  --args -c '<의존성 보정> && exec python /opt/musicnote/mt3_worker.py'
```

**런치 모드가 전부다.** vast는 `--ssh`(그리고 아무것도 안 주면 기본값도) ssh 런타입으로
컨테이너를 만들고 **자기 entrypoint를 넣어 이미지 CMD를 무시한다.** 게다가 그 ssh 주입은
이 최소 이미지(`ubuntu:22.04` 기반)에서 깨져서, 서로 다른 머신 세 대에서 전부

```
Error: remote port forwarding failed for listen port NNNNN
```

가 반복되고 SSH 접속조차 안 됐다. `--args`를 주면 런타입이 `args`가 되어 vast가
**컨테이너를 그대로 실행**한다. `vastai show instance <id> --raw | grep image_runtype`
로 확인할 수 있고, `args`가 아니면 워커는 절대 뜨지 않는다.

`compute_cap<=900` 상한이 중요하다. PyTorch 2.5.1+cu124는 sm_90까지만 지원하고
Blackwell(RTX 50 시리즈)은 sm_120이라 커널이 없다.

`--disk`는 이미지 크기 + 작업 공간이다. 스토리지는 정지 중에도 과금되므로 넉넉하게
잡지 말 것.

### 3. 전환

`gpu.sh up`이 URL을 `deploy/vast/current-url`에 쓰고, `ecosystem.config.js`가
그 파일이 있으면 `MUSICNOTE_MT3_BACKEND=remote`를 켠다. 파일이 없으면 로컬 CPU다.

**pm2는 `restart`로도 `--update-env`로도 이 파일을 다시 읽지 않는다.** 프로세스를
처음 띄울 때 캡처한 환경을 그대로 재생하기 때문이다. `pm2 delete` 후 `pm2 start`만
반영된다 — `gpu.sh`의 `reload_app()`이 그것이다.

`MUSICNOTE_MT3_REMOTE_FALLBACK=1`(기본값)이면 원격 실패 시
`MUSICNOTE_MT3_LOCAL_URL`의 로컬 CPU 워커로 자동 강등된다. GPU가 꺼져 있어도
"느려짐"이지 "고장"이 아니다.

### 4. 쓰고 나면

```bash
vastai stop instance <id>     # GPU 과금 중지, 이미지는 그 머신 디스크에 유지
vastai start instance <id>    # 1분 안쪽
vastai destroy instance <id>  # 스토리지까지 0. 다음엔 이미지를 다시 받는다
```

## 배포 절차 (서버리스)

이미지를 `--target serverless`로 빌드해야 PyWorker가 들어간다. PyWorker는 모델
서버의 로그와 `/health`로 준비 상태를 판단하므로 `start.sh`가 서버를 먼저 띄운다.

템플릿에서 포트·환경변수는 **`env` 필드(Docker options)** 에 넣는다. `args_str`
(컨테이너 인자)에 넣으면 컨테이너가 `-p`라는 프로그램을 실행하려다 죽는다:

```
env     : -p 3000:3000 -p 3001:3001 -e WORKER_PORT=3000
runtype : args
```

엔드포인트 이름을 `musicnote-mt3`로 하면 `MUSICNOTE_VAST_ENDPOINT` 기본값과 맞는다.
`cold_workers` 기본값 5와 `test_workers` 기본값 3은 비용 함정이니 명시적으로 0/1로
누른다.

```bash
MUSICNOTE_MT3_BACKEND=vast
MUSICNOTE_VAST_ENDPOINT=musicnote-mt3
VAST_API_KEY=...
```

## 업데이트

무엇을 고쳤느냐로 갈린다.

| 고친 것 | 해야 할 일 |
|---|---|
| `app.py`, `score_build.py`, `voices.py`, 프런트엔드 … | `pm2 restart musicnote`. GPU 박스와 무관하다 — 거기서는 오디오를 받아 음표를 돌려주는 일만 한다. |
| `backend/mt3_worker.py` | 커밋 후 `gpu.sh down && gpu.sh up`. **재빌드 없음.** |
| `Dockerfile` (의존성, trim) | 이미지를 다시 굽는다. qemu 말고 x86_64에서. |

`gpu.sh`는 컨테이너를 띄울 때 GitHub에서 `mt3_worker.py`를 받아 이미지에 구워진
사본 위에 덮어쓴다. 그래서 **GPU는 워킹트리가 아니라 커밋된 코드를 돈다** — 워커를
고쳤으면 먼저 커밋해야 한다. 받아오기가 실패하면 구워진 사본이 그대로 쓰이므로
GitHub 장애가 기동을 막지는 않는다. 브랜치를 시험하려면 `MT3_WORKER_URL`을 그 ref로
가리키면 된다.

## 이미지가 GPU 박스에서만 깨지는 것들

빌드 호스트는 arm64에 GPU가 없다. 그래서 **이미지 안에서 `import torch`를 해 본 적이
없고**, 아래 셋은 전부 빌린 박스에서야 드러났다. 이미지를 손대면 반드시 실제 워커에서
확인할 것.

| 증상 | 원인 |
|---|---|
| `ImportError: libcupti.so.12` | trim 단계가 `nvidia/cuda_cupti`를 지웠다. 프로파일러 전용이라고 적어놨지만 `import torch`가 즉시 해석한다. 지우면 안 된다. |
| `No module named 'transformers.utils.model_parallel_utils'` | 핀 없이 빌드해서 transformers head를 받았다. YourMT3가 그 모듈을 import한다. |
| `huggingface-hub 1.29.0 ... requires <1.0` | 같은 이유. transformers 4.44가 hub 1.x를 거부한다. |

pip 버전은 CPU 기준선을 만든 `~/mt3-venv/bin/pip freeze`에서 가져와 Dockerfile에
박아 두었다. 핀이 없으면 GPU/CPU note F1 비교 자체가 의미가 없다 — 서로 다른 코드를
돌리게 되기 때문이다.

**qemu 크로스 빌드는 마지막 수단이다.** arm64에서 `/opt/conda` 3.3 GB를 복사하는 데만
10분이 걸리고, 그 레이어 앞에 `ENV` 한 줄을 넣으면 캐시가 깨져 전체를 다시 push해야
한다 (실제로 오늘 그렇게 날렸다). 자주 바뀌는 것은 `mt3_worker.py` 하나뿐이므로
**시작 시 앱 서버에서 그 파일만 받아오게 하면 재빌드가 아예 필요 없다.** 그게 안 되면
x86_64에서 빌드할 것 (GitHub Actions 무료 러너, 또는 vast의 CPU 인스턴스).

## 측정 결과 (2026-08-31, RTX 3060 $0.052/hr)

| | CPU (2코어) | GPU |
|---|---|---|
| 27.9초 클립, MT3 1회 (워밍업 후) | 약 7분 | **7.5초** (추론 5.1초) |
| 27.9초 클립, 앱 전체 경로 (앙상블 2회 + 성부·박자·스템) | 약 15분 | **16초** |
| 첫 요청 (모델 GPU 적재) | — | 25초, 이후 상주 |

정확도는 동일하다. ref05에서 onset F1 CPU 0.979 / GPU 0.973, +offset F1 0.956 / 0.954,
**GPU와 CPU 출력의 상호 일치도 F1 0.994.** 남는 차이는 CUDA 커널의 부동소수점
비결정성이다.

## 검증 (전환 전에 반드시)

```bash
python backend/eval_harness.py --engine api:mt3 eval/refs
```

로컬 CPU 결과와 **note F1이 일치해야** 한다. 기준선은 `ACCURACY.md` 0-2절
(single 0.947, union 0.953). 수치가 다르면 GPU 경로가 다른 모델이나 설정을 쓰고
있다는 뜻이다.

## 보안

- `VAST_API_KEY`는 서버 환경변수/파일에만. 프런트엔드·저장소에 넣지 않는다.
- 업로드 크기 상한은 워커에도 있다 (`MT3_MAX_AUDIO_BYTES`, 기본 48 MB).
- 결과는 notes JSON만 받고, 성부 분리·검증·악보화는 현재 서버에서 한다.

## 참고

- https://docs.vast.ai/guides/serverless/architecture
- https://docs.vast.ai/guides/serverless/creating-new-pyworkers
- https://docs.vast.ai/guides/serverless/workergroup-parameters
- https://docs.vast.ai/sdk/python/serverless/worker-config
