# Vast.ai 서버리스 MT3 (scale-to-zero)

목적: 평소에는 현재 CPU 서버만 돌리고, 정밀 채보가 필요할 때만 GPU를 쓴다.
유휴 시 워커가 0으로 내려가 과금이 없다.

## 실제 구조 (2026-08-30 문서 확인 후 수정)

이전 계획은 "직접 Docker 이미지를 만들어 HTTP 엔드포인트를 노출한다"였는데,
vast 서버리스는 그렇게 동작하지 않는다. **PyWorker** 가 모델 서버 앞에 서서
준비 상태를 판단하고, 요청을 전달하고, 오토스케일러에 부하를 보고한다.

```text
현재 CPU 서버 (musicnote)
  └ mt3_bridge.py  ── vastai SDK ──▶ vast 백엔드가 워커 배정
                                        │
                                        ▼
                         워커 컨테이너 (GPU, 우리 이미지)
                           ├ worker.py       PyWorker (라우팅·부하 보고)
                           └ mt3_worker.py   모델 서버 :8732  ← 기존 파일 그대로
```

핵심: **`backend/mt3_worker.py` 를 그대로 모델 서버로 쓴다.** 로컬 pm2 워커와
같은 파일이며, 원격에서는 `MT3_DEVICE=cuda` 와 base64 오디오 입력만 다르다.

## 코드 쪽 준비 (완료)

| 파일 | 변경 |
|---|---|
| `backend/mt3_worker.py` | `MT3_DEVICE` (cpu/cuda/auto), `audio_b64` 입력, 업로드 크기 제한 |
| `backend/mt3_bridge.py` | `MUSICNOTE_MT3_BACKEND=local\|vast`, 16 kHz mono FLAC 업로드, 원격 실패 시 로컬 폴백 |
| `deploy/vast/worker.py` | PyWorker 설정 (`/transcribe` 핸들러, 오디오 초 단위 워크로드) |
| `deploy/vast/Dockerfile` | CUDA + mt3-infer + 체크포인트 baked-in |
| `deploy/vast/start.sh` | 모델 서버 → PyWorker 순차 기동, 백엔드 죽으면 같이 종료 |

업로드는 원본이 아니라 **16 kHz mono FLAC** 이다. MT3 가 어차피 16 kHz 로
리샘플하므로 원본을 보내면 대부분이 낭비다. 실측: 13 초 클립 2.43 MB → 0.22 MB.

## 배포 절차

### 1. 계정과 키

```bash
pip install vastai
export VAST_API_KEY="..."          # 서버 환경변수에만 둔다. 프런트엔드 금지.
```

### 2. 이미지 빌드·푸시

레지스트리는 vast 워커가 당길 수 있는 공개(또는 인증 설정된) 곳이어야 한다.

```bash
docker build -f deploy/vast/Dockerfile -t <레지스트리>/musicnote-mt3:1 .
docker push <레지스트리>/musicnote-mt3:1
```

체크포인트를 이미지에 굽는 이유: scale-to-zero 워커는 콜드스타트마다 새
컨테이너다. 굽지 않으면 매번 수 GB를 다시 받는다.

### 3. 템플릿 생성

vast 대시보드 → Templates → 위 이미지로 새 템플릿.
- `PYWORKER_REPO` 는 쓰지 않는다 (worker.py 를 이미지에 넣었다).
- 필요 시 `MT3_MODEL`, `MT3_DEVICE=cuda` 등을 템플릿 환경변수로 노출.

### 4. 엔드포인트 + 워커그룹

대시보드 → Serverless. 엔드포인트 이름을 `musicnote-mt3` 로 하면
`MUSICNOTE_VAST_ENDPOINT` 기본값과 맞는다.

권장 초기값:

| 항목 | 값 | 이유 |
|---|---|---|
| `gpu_ram` | 24 GB 이상 | YourMT3 CPU 피크가 7.5 GB, GPU 여유 확보 |
| min workers | 0 | scale-to-zero |
| max workers | 1 | 개인 사용, 비용 상한 |
| `template_hash` | 위 템플릿 | 커스텀 이미지 지정 경로 |

### 5. 전환

```bash
# ecosystem.config.js 의 musicnote 프로세스 env 에 추가
MUSICNOTE_MT3_BACKEND=vast
MUSICNOTE_VAST_ENDPOINT=musicnote-mt3
VAST_API_KEY=...
```

`MUSICNOTE_MT3_REMOTE_FALLBACK=1` (기본값) 이면 원격 실패 시 로컬 CPU 워커로
자동 강등된다. GPU 장애가 "느려짐"이 되지 "고장"이 되지 않는다.

### 6. 검증 (전환 전에)

```bash
python backend/eval_harness.py --engine api:mt3 eval/refs
```

로컬 CPU 결과와 **note F1 이 일치해야** 한다. 기준선은 `ACCURACY.md` 0-2 절.
수치가 다르면 GPU 경로가 다른 모델/설정을 쓰고 있다는 뜻이다.

## 콜드스타트와 비용

- 유휴 시 과금 0. 대신 **첫 요청은 컨테이너 기동 + 모델 로드로 수 분** 걸린다.
- 앙상블이 기본이면 요청당 MT3 추론이 2회다 (`MUSICNOTE_MT3_ENSEMBLE_SHIFT`).
  같은 워커가 연속 처리하므로 콜드스타트는 한 번만 문다.
- 현재 CPU 는 실시간의 약 13배다. 13 초 클립이 약 3분, 앙상블이면 6분.

## 보안

- `VAST_API_KEY` 는 서버 환경변수에만. 프런트엔드·저장소에 넣지 않는다.
- 업로드 크기 상한은 워커에도 있다 (`MT3_MAX_AUDIO_BYTES`, 기본 48 MB).
- 결과는 notes JSON 만 받고, 성부 분리·검증·악보화는 현재 서버에서 한다.

## 참고

- https://docs.vast.ai/guides/serverless/architecture
- https://docs.vast.ai/guides/serverless/creating-new-pyworkers
- https://docs.vast.ai/guides/serverless/workergroup-parameters
- https://docs.vast.ai/sdk/python/serverless/worker-config
