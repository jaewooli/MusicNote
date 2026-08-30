# Vast.ai 서버리스 MT3 전사 계획

목적: 평소에는 현재 CPU 서버만 실행하고, 사용자가 전체 악기 채보를 요청할 때만 Vast GPU를 사용한다.

## 권장 구조

```text
MusicNote API (현재 서버)
  -> WAV bytes 또는 signed object URL
  -> Vast Serverless MT3 endpoint (GPU)
  -> notes JSON
  -> 성부 분리 · 품질 검증 · 악보 생성 (현재 서버)
```

현재 `mt3_bridge.py`는 원격에 존재하지 않는 로컬 `wav_path`를 전송하므로, 먼저 **WAV 업로드형 요청**으로 변경해야 한다. 단순히 `MUSICNOTE_MT3_URL`만 Vast 주소로 바꾸면 동작하지 않는다.

## 서버리스 권장 초기값

- GPU: 1개, VRAM 24GB 이상(3090/4090/A5000급), CUDA 12+
- `max_workers=1`: 모델 메모리와 긴 단일 전사 작업을 고려
- `min_load=0`, `cold_workers=0`: 유휴 시 완전 scale-to-zero
- `inactivity_timeout=600`: 마지막 작업 10분 뒤 워커 종료
- 큐 시간: `target_queue_time=30`, `max_queue_time=180`
- 모델 체크포인트는 컨테이너 이미지 또는 영속 볼륨에 둔다. 첫 요청은 모델 로딩 cold start가 수 분 걸릴 수 있다.

## 단계

1. Vast 계정·API 키·크레딧을 준비한다.
2. MT3 전용 Docker 이미지를 만든다. 입력: WAV bytes, 출력: `notes`, `tracks`, `model` JSON.
3. Vast Serverless endpoint와 worker group을 만든다. 24GB VRAM·CUDA 버전·신뢰도 필터를 설정한다.
4. `mt3_bridge.py`에 원격 업로드 클라이언트를 추가한다. API 키는 현재 서버의 환경변수에만 보관한다.
5. 짧은 WAV로 cold/warm latency, 음표 F1, 비용을 측정한다.
6. 통과 후 `MUSICNOTE_MT3_BACKEND=vast` 같은 명시적 설정으로 전환한다. 로컬 워커는 fallback으로 유지한다.

## 보안

- Vast API 키를 프런트엔드에 넣지 않는다.
- MT3 endpoint는 인증된 서버 간 호출만 허용한다.
- 업로드 파일 크기·형식·작업 시간 제한을 endpoint에도 적용한다.
- 결과 notes JSON만 받아 현재 서버에서 검증·악보화한다.

## 참고

- https://docs.vast.ai/guides/serverless
- https://docs.vast.ai/guides/serverless/managing-scale
- https://docs.vast.ai/guides/serverless/serverless-parameters
