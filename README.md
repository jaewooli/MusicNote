# MusicNote

음원 파일(mp3/wav/flac/m4a/ogg …) 또는 **YouTube 링크**에서
**멜로디와 각 음(note)** 을 추출하는 웹 서비스 프로토타입.

- 파일 업로드 또는 YouTube URL → 진행률 표시 → 피아노 롤 시각화 + 음 목록 표 + 피치 곡선
- 결과를 **MIDI** / **JSON** 으로 다운로드
- 원본 음원 + **추출 결과 합성음**, 둘 다 일반 `<audio controls>` 로 재생 — 합성음은
  `OfflineAudioContext` 로 WAV 로 미리 렌더해 blob 으로 넣으므로 모바일에서 **언락 절차
  없이** 재생 버튼만 누르면 소리가 난다 (iOS 무음 스위치 영향도 안 받음). 배음 3개를
  더한 톤 + **피크 정규화**로 멜로디(단선율)도 다성만큼 크게 들린다.
- **민감도 슬라이더**로 검출 정확도/개수를 실시간 조정 (재분석 없이 ≈10 ms)
- 진행률은 **단계별**(다운로드 / 분석)로 각 0~100%, `(1/2)` 식 단계 표시

## 구조

```
backend/
  app.py              FastAPI 앱 (업로드/분석/다운로드 엔드포인트)
  transcribe.py       음 추출 로직 (엔진 3종)
  requirements.txt    핵심 의존성
  requirements.lock.txt  실제 설치된 전체 버전 스냅샷
frontend/
  index.html          단일 파일 프론트엔드 (바닐라 JS + canvas)
ecosystem.config.js   pm2 프로세스 정의
deploy/
  musicnote.nginx.conf  nginx vhost (musicnote.ddyoru.cloud)
setup.sh              환경 구축 스크립트
uploads/              임시 작업 디렉터리 (30분 후 자동 정리)
logs/                 pm2 로그
```

## 분석 엔진

| 모드 | 엔진 | 설명 |
|------|------|------|
| `melody` | **pYIN** (librosa) | 단선율. 노래·독주에 적합. 항상 사용 가능. |
| `polyphonic` | **basic-pitch** (Spotify, ONNX) | 화음 포함 전체 채보. 설치되어 있으면 자동 사용. |
| `polyphonic` (대체) | **CQT 피크 추출** | basic-pitch 미설치 시 자동 폴백. 정확도 낮음. |

`GET /api/health` 의 `basic_pitch` 필드로 현재 어떤 엔진이 활성인지 확인.

### 항상 적용되는 정확도 보정 (민감도와 무관)

- **전역 튜닝 추정** (`librosa.estimate_tuning`) 을 반음 반올림 전에 빼서, A≠440 이나
  약간 낮게/높게 부른 음원에서 반음 어긋남을 줄인다.
- **피치 곡선 메디안 평활 + 옥타브 점프 교정** — 1~2 프레임 튐 제거, 국소 음역에서
  ±한 옥타브 벗어난 값을 되돌린다. (`transcribe._smooth_midi_track`)
- **온셋 기준 음 경계** — `librosa.onset.onset_detect` 로 음 시작을 스냅하고, 같은
  음높이라도 새 어택이 있으면 음을 나눈다.
- polyphonic: **옥타브 중복 제거** — 더 큰 음 위 정확히 +12 이면서 대부분 겹치고 더
  여린 음은 배음으로 보고 버린다.

### 악기(음색) 분류 → 채보 프리셋

순수 피치만 보면 **친 소리(피아노·기타)**와 **켠 소리(바이올린 등)**를 똑같이 다뤄서
둘 다 틀린다(피아노: 어택 놓침·음 과도하게 유지 / 바이올린: 비브라토가 여러 음으로 쪼개짐,
여린 진입 누락). `transcribe.classify_instrument` 가 파형 특징으로 지배적 음색을 분류한다:

- 특징: **하모닉/퍼커시브 에너지비**(HPSS), 스펙트럼 **중심주파수·평탄도**, ZCR,
  **시간 중심**(에너지가 앞쏠림=타현/발현, 중간=지속음), 온셋 밀도 — 가장 큰 ~25초 구간에서.
- → 프리셋: `struck`(타현·발현) / `sustained`(활·관·성악) / `percussive`(타악) / `neutral`.
  프리셋이 멜로디: **평활 커널·온셋 가중치·최소 음길이·병합 간격**을, polyphonic:
  **basic-pitch onset/frame 임계 + 최소 길이 + melodia trick**을 바꾼다.
- 결과 화면 **악기 드롭다운**에서 자동 감지값을 확인하고 수동으로 바꿀 수 있다
  (`POST /api/refine` 에 `instrument` 전달, 민감도와 마찬가지로 재분석 없이 즉시).

### 민감도(정확도) 실시간 조정

무거운 분석(pYIN·basic-pitch)은 **한 번만** 돌리고 그 중간 산출물을 `job_id` 로 캐시한다
(`transcribe.analyze`). 결과 화면의 **민감도 슬라이더**를 드래그하면 `POST /api/refine/{job_id}`
가 **세그먼트 단계만** 다시 돌린다(≈10 ms). 재분석·재다운로드 없음.

- **낮음** → 임계값 높임 = 확실한 음만 (놓침 ↑, 오검출 ↓)
- **높음** → 임계값 낮춤 = 작은·짧은 음까지 (오검출 ↑, 놓침 ↓)
- melody: pYIN 유성확률 임계 + 최소 음길이. polyphonic: basic-pitch posteriorgram 의
  onset/frame 임계 + 최소 길이를 다시 적용(캐시한 raw posteriorgram 재활용).
- 슬라이더를 움직이면 표·피아노 롤·MIDI·합성음 재생이 모두 갱신된다.
- 캐시(=조정 가능 기간)는 작업 후 30분.

### `stems` 모드 — 악기별 분리 채보 (Demucs, 선택 설치)

곡을 악기군별 스템으로 나눠 **각 스템을 따로 채보**한다. 요청: "어느 악기가, 곡의 어느
구간에서, 어떤 선율을 연주하는지".

1. **Demucs `htdemucs_6s`** → drums / bass / vocals / guitar / piano / other 6 스템.
   `stems.separate()` 가 각 스템을 22 kHz 모노로 저장하고 RMS 포락선으로 **등장 구간**
   (`spans`)과 **점유율**(`presence`)을 뽑는다. 무음/누출 스템은 버린다.
2. 스템 → **MuseScore 악기군** 매핑: drums→타악, bass→베이스, vocals→성악,
   guitar→발현 현악, piano→건반. `other` 스템만 `classify_instrument` 로
   찰현 현악 / 목관 / 신스 등으로 세분한다.
3. 스템별 채보 — **검출 단계를 학습된 모델로**:
   - 단선율 스템(베이스·성악·현악·관악·melody 모드의 리드) → **torchcrepe** (CNN,
     `MUSICNOTE_CREPE_MODEL=tiny`; `full` 은 이 2-CPU 박스에서 ~7배 realtime·OOM 위험).
     pYIN 은 미설치 시 폴백.
   - 피아노 스템 → **piano_transcription_inference** (Kong et al., MAESTRO F1 0.97
     체크포인트, `~/piano_transcription_inference_data/`). 미설치 시 basic-pitch.
   - 기타/other 화음 스템 → basic-pitch.
   - 악기군별 pYIN/CREPE **주파수 범위**(`_FAMILY_RANGE`, 베이스 31–400 Hz), `bass`
     전용 세그먼트 프리셋, 스템별 튜닝 추정, blip 제거, melody 리드는 tessitura 밖
     이상치 제거.
   - 스템별 MIDI(General-MIDI program) + 스템 오디오 재생.
4. **박자 양자화** (`quantize`): `librosa.beat` 로 곡 전체 비트를 잡아 음 시작·길이를
   16분음표 격자에 스냅. `POST /api/refine` 의 `quantize:true`, 프론트 "박자 정렬" 체크.
5. 각 스템도 `POST /api/refine/{job_id}` 에 `stem` 을 실으면 민감도·악기·양자화를
   **실시간 재조정**(스템별 분석 캐시 재사용).

**정확도 측정**: `python backend/eval_harness.py MODE ref.mid audio.wav …` 가
mir_eval 로 note/onset F1 을 낸다. MAESTRO/URMP 몇 곡을 참조로 두면 변경이 도움이
되는지 눈대중 없이 확인 가능.

**분리 누출(ghost note) 게이트** — Demucs 스템에서만 켠다 (`app._analyze_stem` 이
`a["gate"]=True`, `a["spans"]` 주입). `transcribe._gate_notes` 가 세그먼트 후 두 단계로
거른다: (1) **무음 구간 게이트** — 스템 자신의 RMS 활성 구간(`stems._active_spans`)과
겹침이 부족한 음 제거, (2) **배음 salience 게이트** — 스템 자신의 constant-Q 맵
(`_salience_cqt`, 분석 시 캐시)에서 그 음의 기음/2·3배음 에너지가 같은 순간 전체 대비
너무 약하면 제거. 다성 스템은 훨씬 약하게(하위 15%·낮은 floor), 단선율은 하위 30%.
음이 6개 미만이면 미동작, `refine` 은 여전히 ~10 ms (캐시 재사용).

**음악적 후처리** (`_musical_cleanup`, 항상 켜짐·보수적) — 세그먼트 후 순서대로:
① **옥타브 정합**(단선율): ±1.5 s 창 중앙값 대비 정확히 ±12/±24 벗어난 음을 되돌림.
② **조성 스냅**: Krumhansl–Schmuckle 로 `_sal` chroma 에서 조 추정(`_key_from_sal`),
   strength ≥ 0.6 이고 조 밖이며 *짧거나 양옆이 조 안*인 음만 ≤1 반음 이동(장조 온음계 +
   자연단조 + 상행 7음 허용). ③ **박자 시작 스냅**: `mix_beats` 16분 격자에 40 ms 이내면
   흡착(루바토 보존). ④ **초단음 제거**: 온셋 지지 없는 32분음표 미만 blip 제거.

**voiced 히스테리시스** (`segment_melody`): 음을 *시작*하려면 확률이 높은 문턱을,
*유지*는 낮은 문턱을 넘으면 된다(Schmitt) → 잡음 blip·중간 끊김 동시 감소. `_segment`
가 0/1/2 레벨 배열을 받게 확장(bool 마스크는 그대로 동작).

**음표 세기·엔벨로프** (`_note_dynamics`, `refine` 상시) — 각 음의 실제 소리 크기와 모양을
그 음 **자신의** constant-Q 행(기음+2·3배음) 시계열에서 읽는다. `velocity` = 전체 대비
상대 크기(√ 압축), `env` = 10-포인트 0-127 진폭 곡선(자기 피크 정규화). 멜로디 엔진(가짜
velocity 90)은 전부 덮어쓰고, 다성 엔진은 모델 velocity 와 6:4 블렌드. 프론트 `Player._voice`
가 `env` 를 `setValueCurveAtTime` 게인 곡선으로 적용 → "한 번 치고 감쇠", "점점 커짐" 등이
실제로 들린다. **MusicXML**(`musicxml.py`)은 읽기 쉽게: 연속 3음↑ 한 방향으로 12 이상
변하면 **크레셴도/디미누엔도 머리핀**(`<wedge>`), 그 밖의 계단식 변화(≥12)에만
`<direction><dynamics>`(pp~ff) + `<sound dynamics>`. 머리핀 구간 안쪽은 기호 생략(악보
도배 방지). 음마다의 정확한 세기·`env` 는 JSON 과 재생에 그대로 남는다.

**음표 신뢰도** (`_annotate_confidence`) — 각 음에 `conf` 0..1: 검출기 자체 확신
(crepe periodicity / basic-pitch note posterior / velocity)과 배음 salience 의 6:4 혼합.
`result.low_conf` = conf<0.5 개수, `result.key` = 추정 조성. UI: 표 "신뢰" 열,
피아노 롤 빨강 점선·노랑 테두리, 악보 주황빛 음표, 악보 조표는 감지 조성을 자동 반영.

**CREPE 개선** — 기본 디코더 `viterbi`(시간축 디코딩 → 옥타브 점프 급감, tiny 에서 비용
거의 0: 측정 0.46× RT). `_yin_octave_correct` 가 YIN 과 5프레임 이상 옥타브 불일치인
구간만 CREPE 를 YIN 옥타브로 스냅. `full` 은 이 박스에서 ~8× RT — 기본 비활성,
`MUSICNOTE_CREPE_MODEL=full` 또는 `MUSICNOTE_CREPE_FULL_MAX` 로 opt-in.

**하모닉 NMF 앙상블** (`_harmonic_nmf` / `_nmf_ensemble`, basic-pitch·CQT 폴백 전용) —
캐시된 `_sal` CQT 를 **고정된 per-pitch 하모닉 콤 사전**(W 고정, 활성 H 만 25회 곱셈갱신,
A1–C7 64음)으로 분해해 basic-pitch 와 독립적인 pitch×frame 활성맵을 얻는다. `refine` 에서:
① NMF 지지가 없는 *조용한* basic-pitch 음 제거(민감도 낮을 때만, ≤25%),
② 어떤 음도 안 덮는 지속 활성을 음으로 **회수**(민감도 ≥ 0.4, 상한 있음, 옥타브 위가
더 강하면 sub-harmonic 유령으로 보고 skip). 회수 음은 `_nmf_added` 표시 + conf ≤ 0.6.
`_annotate_confidence` 는 NMF 지지를 conf 혼합에 25% 반영(det 45 / salience 30 / nmf 25).
비용: W.T·W·H 가 64²·frames·25 ≈ 1초 미만, `refine` 은 여전히 가볍다.

### 정확도 측정 (`eval/`) — 튜닝의 유일한 근거

`eval/build_refs.py` 가 **MAESTRO**(실제 연주자의 Disklavier 녹음 = 완벽한 ground truth)
MIDI에서 음이 가장 조밀한 구간을 잘라 fluidsynth로 렌더 → `(ref*.wav, ref*.mid)` 쌍 생성.
`backend/eval_harness.py` 가 mir_eval로 채점한다:

```bash
python eval/build_refs.py --n 6 --secs 25          # 참조 세트 생성
python backend/eval_harness.py eval/refs --engine direct:polyphonic
python backend/eval_harness.py eval/refs --engine api:mt3     # 실제 파이프라인
```

세 지표: **onset F1**(50 ms) · **note F1**(+음정, AMT 표준) · **note+offset F1**(+길이 20%).
`est/ref` 음 수도 같이 찍어 과검출·누락을 구분한다.

> 사운드폰트 렌더는 실제 녹음보다 **쉬운** 조건이라 절대값은 낙관적이다. 목적은
> **상대 비교**(변경이 도움이 됐는가)이므로 그것으로 충분하다.

**2026-08-30 최초 측정 (6클립, 피아노):** `basic-pitch` onset F1 **0.717** / note F1 **0.526** /
**note+offset F1 0.086**, est/ref 412/547. → 음 위치는 그럭저럭인데 **음 길이가 사실상 노이즈**이고
30~40% 누락. 민감도를 0.8로 올리면 1149음 과검출에 note F1은 오히려 0.448로 **악화** — 즉
민감도 노브로 해결되는 문제가 아니다. 이 숫자가 아래 악보 재설계의 근거다.

### 악보 문서 모델 (ScoreDoc) — 표기의 단일 진실

예전에는 화면(JS `buildMeasures`)과 내보내기(`musicxml.py`)가 **각자** 음표 리스트에서
악보를 재유추했다 → 16분음표 격자 고정, **각 지점 최고음만**(화음 소멸), 셋잇단 불가,
둘이 서로 어긋남. 이제:

- **`backend/score_model.py`** — `ScoreDoc → Part → Voice → Measure → Chord/Rest → Note`.
  시간은 초가 아니라 **틱**(4분음표당 480). 음가는 `type`+`dots`+`tuplet`으로 명시.
  `split_duration()`(붙임줄 분해), `spell()`(조표에 맞춘 임시표), `krumhansl_fifths()`(조 추정).
- **`backend/score_build.py`** — `build_score(parts, beats, tempo, time_sig)`: ① 비트 그리드
  구성(감지 비트 없으면 템포로 외삽) ② 박자표로 마디 분할 ③ **박마다 분할을 독립 결정**
  (2/4/8분 vs 셋잇단 — 오차 최소인 쪽, **파트별로** 판단하므로 베이스 셋잇단이 피아노를
  오염시키지 않음) ④ 온셋/오프셋 양자화 ⑤ **화음·쉼표·마디 넘는 붙임줄·셋잇단 괄호** 생성.
  표기 위생: 겹점 금지(`MAX_DOTS=1`), 32분음표 미만 금지, 16분 미만 빈틈은 앞 음이 흡수,
  표기 불가능한 길이는 **가장 가까운 읽을 수 있는 음가로 스냅**(음 끝이 노이즈이므로
  정확하지만 못 읽는 값보다 낫다).
- **소비자는 모두 이 문서를 렌더**한다: `musicxml.py` 는 **직렬화 전용**(`doc_to_musicxml`),
  프론트는 `GET /api/score/{job}` 로 같은 ScoreDoc을 받아 `renderDoc()` 으로 그린다.
  → 화면과 내보내기가 구조적으로 어긋날 수 없다.

`GET /api/score/{job_id}?stem=&num=&den=&tempo=&fifths=` → ScoreDoc JSON.
사용자가 템포를 바꾸면 감지 비트 배열은 더 이상 유효하지 않으므로 자동으로 버리고
템포 기반 그리드를 쓴다.

### 프론트 구조 — 두 페이지

`frontend/` 는 공유 코어 + 두 페이지:
- **`musicnote-core.js`** — 공유 로직 전부: 헬퍼, `Score`(VexFlow 렌더), `Editor`(피아노
  롤 편집), `drawRoll`, 합성음, 스템 렌더, `renderCommon()`, 그리고 **`MN`**(작업 영속).
  모든 `$('#…')` 배선은 요소 존재 여부를 확인해 두 페이지 모두에서 안전.
- **`index.html`** — 분석 페이지: 업로드·진행바·통계·원본/합성음·민감도 슬라이더·스템
  분해·피아노 롤(보기 전용)·**악보 미리보기**(감지 조성/4-4/추정 템포)·음 목록. 편집
  UI 없음. `render()` = 통계+오디오+refine 블록 + `renderCommon()`.
- **`editor.html`** — 악보 편집 전용 페이지. `editor.html?job=<id>[&stem=<sid>]` 로 진입.
  풀 악보 툴바(조표·박자표·템포·💾 저장·MusicXML·MIDI편집본·JSON), 항상 켜진 편집기,
  전송(재생) 바, 음 목록, 스템 선택기. 분석 페이지 "✏️ 악보·리듬 편집기 열기" 버튼
  (`#openEditor`)이 현재 스템까지 실어 링크.
- **편집 = 피아노 롤 + 악보 둘 다** (`Editor` + `scoreBind()`):
  - 피아노 롤: 노트 클릭=선택(Shift 다중), 본체 드래그=이동, 오른쪽 끝=길이, 더블클릭/Del=삭제,
    ＋그리기, 격자 스냅(1/4·1/8·1/16·8분셋), Ctrl+Z/Y(스택 60), ↑↓ 반음/Shift 옥타브, 원래대로.
  - **악보**: 음표 클릭=선택(롤과 연동, `Score._hit` 로 힛테스트 — 각 StaveNote `getBoundingBox()`),
    세로 드래그=음정 이동(≈6 px/반음), 더블클릭=삭제, ＋그리기+빈 곳 클릭=새 음
    (`Score.yToPitch()` 로 y→조성에 맞춘 온음계 음, x→마디·격자). 선택된 음은 악보에서 파란색.
- **저장/내보내기**: `POST /api/edit/{job_id}` `{notes,tempo,time_sig,title}` → `{job_id}.mid`
  재작성 + `{job_id}.musicxml` 생성(`backend/musicxml.py`). 💾 버튼 또는 편집 시 자동
  (디바운스 450 ms). `GET /api/download/{id}.musicxml` 로 MuseScore/Finale 에서 열기.
- **재생 = 라이브 Web Audio 룩어헤드 스케줄러** (`Player`, in core): OfflineAudioContext→WAV
  렌더링 폐기. `Player.setNotes()` 즉시(스템/민감도 바꿔도 재-렌더 없음). `_begin()` 이
  먼저 `ctx.resume()` 를 기다린 뒤 40 ms 폴링(`_pump`)으로 음을 ~0.3 s 앞서 스케줄 →
  "삐빅 한 번 나고 무음" 버그 해결(예전엔 resume 전에 전 구간 오실레이터를 큐잉해 엔벨로프가
  전부 과거로). 전송 바(▶/⏸·시크·시간), Space 토글.
- **재생 헤드**(`Playhead`): `#rollHead`(롤)·`#scoreHead`(악보)를 `Player` 프레임마다 이동,
  재생 중 자동 스크롤. 악보 시간→x 는 `Score._layout`(마디별 x·w·y·tStart·tEnd). 롤/악보
  모두 `#rollInner`/`#scoreInner`(inline-block)로 감싸 헤드가 스크롤과 함께 움직임.

### `score.html` — 전체 악보 (악기별, 전체 화면)

`score.html?job=<id>` 로 진입. `renderFullScore(box, parts, opts)` (in core) 가 **악기(스템)
마다 한 단씩** 세로로 쌓아 시스템 단위로 그린다(왼쪽 브래킷 `StaveConnector`). 결과에 스템이
있으면 `pitched && notes` 스템들이 `parts`, 없으면 단일 파트. 조표/박자표/템포 툴바로 재렌더.
**재생**: `Player.setNotes(모든 파트 합침)`, 상단 전송 바(▶/⏸·시크·시간, Space 토글).
`renderFullScore` 가 `Score._layout`(마디별 x·w·시스템 y·h·tStart·tEnd)을 채우고 `#scoreHead`
빨간 재생 헤드가 **모든 악기 단을 가로질러** 이동, 재생 중 가로·세로 자동 스크롤.
`@media print` 로 헤더·툴바·재생 헤드 숨기고 흰 배경 → 🖨 인쇄 그대로 악보. 분석 페이지·
편집기에 "🖥 전체 악보" 링크(`#openFullScore`).

### 새로고침해도 재분석 안 함 (작업 영속)

`MN` (in core): 분석 제출 시 `job_id` 를 `localStorage['mn_job']` + `location.hash`
(`#job=<id>`) 에 기록. 페이지 로드 시 `resume()` 이 hash → localStorage 순으로 id 를 찾아
`GET /api/progress/{id}` 호출: `done` → `showResult()` 로 결과 복원(재분석 없음),
`running` → 진행바 + `pollJob()` 재개, `404`(만료) → 클리어. 서버 `RESULT_TTL` 1 h,
`refine`/`edit` 마다 `_keep_alive()` 로 연장. "새 분석" 버튼(`#newJob`)이 클리어.

### `mt3` 모드 — 학습된 다악기 채보 (MT3 / YourMT3)

`mt3-infer`(PyPI, PyTorch 전용, `magenta/mt3` 의 JAX/t5x/tensorflow 스택을 피하려고
만들어진 툴킷)를 **격리된 별도 venv**(`~/mt3-venv`)에서 워커로 돌린다. 곡 전체를 한 번에
악기별 트랙으로 받아 적음 — Demucs 분리·후처리 없이.

- **빌드**: `deploy/mt3-setup.sh [yourmt3|mr_mt3]` → venv + CPU torch 2.4.1 + mt3-infer +
  **`transformers==4.44.2`**(5.x 는 torch≥2.5 요구+import 버그, ≥4.45 는 YourMT3 디코더가
  `NoneType` 에러) + 벤더된 MR-MT3 `t5.py` 의 `past_key_values=`→`past_key_value=` 한 줄
  패치 + 체크포인트 다운로드. **버전 조합이 예민하니 임의로 올리지 말 것.**
- **워커**: `backend/mt3_worker.py`, pm2 앱 `mt3-worker`(127.0.0.1:8732). 지연 로드 +
  `MT3_IDLE_UNLOAD` 초 후 모델 해제. `POST /transcribe {wav_path}` → 음표 JSON(음마다
  GM program·is_drum·track). `backend/mt3_bridge.py` 가 메인 venv 쪽 클라이언트.
- **app**: `mode="mt3"` → Demucs 생략, 워커 호출, GM program → `mt3_bridge.map_family()`
  로 MuseScore 악기군 매핑 → `stems` 형태 결과로 조립 → 기존 스템 UI·악보·편집기 재사용.
  `_MT3_SLOT` + `_STEMS_SLOT` 을 둘 다 잡아 Demucs 와 절대 동시 실행 안 함. `refine` 은
  캐시된 raw 음표를 velocity/length 로 재필터(가벼움).
- **모델 선택** (실측, 20초 실곡, 2-CPU):
  | | 추론 | RAM | 트랙 | 비고 |
  |---|---|---|---|---|
  | **YourMT3** (기본) | ~7.4× 곡 길이 | **~7.5 GB** | 7 (피아노·EP·기타·베이스·현·색소폰·드럼) | 품질 좋음. `MT3_IDLE_UNLOAD` 로 유휴 시 해제 |
  | MR-MT3 | ~11× 곡 길이 | ~0.7 GB | **2** (피아노+드럼만) | 가볍지만 다악기 분리 빈약 → 비권장 |
- **한계**: GPU 없는 이 서버에서 곡당 **곡 길이의 7배 이상** 시간(3분 곡 ≈ 20분+),
  YourMT3 는 RAM 7.5 GB 라 MT3 작업 중엔 다른 무거운 작업 불가. opt-in 이며 UI 에 경고.
- `pip install git+https://github.com/magenta/mt3`(JAX 원본)은 여전히 불가: `t5x` 가
  aarch64 휠 없는 `tensorflow-cpu` 를 하드 요구 + Python 3.12 이전 핀. Omnizart 도 madmom
  빌드 실패로 불가.

**모든 모드가 Demucs 를 거친다** (`S.available()` 이면): `melody` = 분리한 리드 스템
(vocals 우선, 없으면 저음이 아닌 스템 중 음역·음수 최고) 한 줄, `polyphonic` = 모든
스템 채보를 **하나로 합쳐** 음마다 `stem`/`inst` 태그(피아노 롤은 스템별 색), `stems` =
스템별로 나눠 보기. demucs 미설치 서버만 raw 믹스에서 바로 채보한다.

응답: `result.stems[]` (id, family, label, spans, presence, notes, contour, midi_url,
audio_url, engine). 최상위 `notes`/`midi_url` 은 가장 음이 많은 스템을 미러링(기존 UI 호환).

- **비용**: torch+demucs ≈ 1.5 GB, CPU 에서 실시간의 ~3배(15초 클립 ≈ 40초).
  이 서버 상한 `MUSICNOTE_STEMS_MAX_DURATION=300`(5분), 동시 1건, pm2 `max_memory_restart`
  를 4 GB 로 올림. `demucs>=4.1` 은 Rust 빌드가 필요 → **4.0.1 로 고정**.
- 미설치 서버에서는 `/api/health` 의 `stems:false`, 프론트에서 이 모드가 비활성.

## 로컬 실행

```bash
./setup.sh                        # ffmpeg + venv + 의존성 (basic-pitch 실패해도 계속)
.venv/bin/uvicorn app:app --app-dir backend --port 8731
# http://localhost:8731
```

## pm2

```bash
pm2 start ecosystem.config.js     # 127.0.0.1:8731 에서 기동
pm2 logs musicnote
pm2 restart musicnote
pm2 save                          # 부팅 시 자동 복구 목록에 저장
```

부팅 자동 시작은 `pm2 startup systemd -u ubuntu --hp /home/ubuntu` 로 설치됨(설치 완료).

## nginx (musicnote.ddyoru.cloud)

**배포 완료 — https://musicnote.ddyoru.cloud 로 접속 가능.**

이 호스트는 `*.ddyoru.cloud` 와일드카드 DNS + 공유 멀티-SAN 인증서 구조를 쓴다.
따라서 서브도메인용 A 레코드는 따로 필요 없었고(와일드카드로 이미 해석됨),
아래 작업만 수행했다.

1. `deploy/musicnote.nginx.conf` → `/etc/nginx/sites-{available,enabled}/musicnote`
   (443 ssl + `snippets/proxy-common.conf` + `127.0.0.1:8731` 프록시, `client_max_body_size 48m`)
2. 공유 인증서(`canary.ddyoru.cloud`)에 `musicnote.ddyoru.cloud` SAN 추가:
   ```bash
   sudo certbot certonly --nginx --expand --cert-name canary.ddyoru.cloud \
     -d canary.ddyoru.cloud -d gatesimulator.ddyoru.cloud -d homepage.ddyoru.cloud \
     -d ossimulator.ddyoru.cloud -d pizza.ddyoru.cloud -d privacy-ai-api.ddyoru.cloud \
     -d privacy.ddyoru.cloud -d musicnote.ddyoru.cloud
   sudo systemctl reload nginx
   ```
   `certonly` 라 기존 nginx 설정은 건드리지 않는다. 자동 갱신 시 8개 도메인 전체가 함께 갱신된다.

포트 80 → HTTPS 리다이렉트는 기존 전역 `default_server` 가 처리한다.

검증:
```bash
curl -s --resolve musicnote.ddyoru.cloud:443:127.0.0.1 \
  https://musicnote.ddyoru.cloud/api/health   # -> {"status":"ok",...}, 인증서 검증 통과
```

## API

분석은 **비동기 작업**이다: `POST /api/transcribe` 가 즉시 `job_id` 를 주고(202),
클라이언트가 `GET /api/progress/{job_id}` 를 폴링해 진행률을 받고, 완료 시 결과가 실린다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 상태 + 사용 가능 엔진 |
| POST | `/api/transcribe` | multipart: `file` **또는** `url`(YouTube), `mode`(`melody`\|`polyphonic`\|`stems`) → `{"job_id": "..."}` (202) |
| GET | `/api/progress/{job_id}` | `{status, stage, pct(0~1, 단계 내 진행률), message, step, steps}`; `status=="done"` 이면 `result`, `"error"` 면 `error`+`http` |
| POST | `/api/refine/{job_id}` | body `{sensitivity(0~1), instrument?, stem?}` → 재분석 없이 재-세그먼트(≈10 ms). `stem` 지정 시 stems 모드의 해당 스템만 갱신(`changed_stem` 반환) |
| GET | `/api/download/{job_id}.mid` | 분석 결과 MIDI (30분 유효) |
| GET | `/api/audio/{job_id}.<ext>` | 분석에 쓴 원본 오디오 (브라우저 재생용, 30분 유효) |

`stage`: `download`(URL만, step 1) `→ analyze`(step 2, 파일이면 step 1). 각 단계에서 `pct`
는 그 **단계 내부** 0→1 로 진행한다(다운로드는 실제 바이트 비율, 분석은 오디오 길이 기반
추정 크리프). `step`/`steps` 로 `(1/2)` 식 표시.

`/api/progress` 완료 응답의 `result` 예:
```json
{
  "engine": "pyin", "mode": "melody",
  "duration": 2.2, "tempo": 0.0, "note_count": 4, "sensitivity": 0.5,
  "instrument": {"selected":"auto","detected":"piano","detected_label":"피아노",
                 "preset":"struck","options":[{"value":"auto","label":"자동 감지 (피아노)"}, "..."]},
  "notes": [{"start":0.012,"end":0.557,"pitch":60,"name":"C4","freq":261.63,"velocity":90}],
  "contour": [{"t":0.03,"freq":261.5,"midi":59.98}],
  "midi_url": "/api/download/<id>.mid",
  "audio_url": "/api/audio/<id>.wav",
  "job_id": "<id>", "filename": "노래 제목 (YouTube)"
}
```

## YouTube 링크 입력

`url` 필드에 YouTube 링크를 주면 서버가 `yt-dlp` 로 오디오를 받아 동일하게 분석한다.
`watch?v=`, `youtu.be/`, `shorts/`, `live/`, `embed/` 형식 지원. 최대 20분(`MUSICNOTE_MAX_DURATION`).

**2026-08 현재 동작함.** 현재 YouTube 다운로드에 필요한 3가지가 모두 갖춰져 있다:

### 1. 로그인 쿠키 (`backend/cookies.txt`)

데이터센터 IP는 익명 요청이 봇 차단된다. **전용(버리는) 구글 계정**의 로그인 쿠키가 필요하다.
현재 `jaewolee2022@gmail.com` 계정 쿠키가 설치되어 있다 (개인 계정 아님).

- 재발급: `deploy/yt_login.py` 를 쓰면 브라우저 export 확장앱 없이 헤드리스로 뽑을 수 있다.
  ```bash
  xvfb-run -a .venv/bin/python deploy/yt_login.py <email> <password> backend/cookies.txt
  # 2단계 인증(기기 승인) 프롬프트가 뜨면 폰에서 숫자 승인 → 자동으로 쿠키 저장
  chmod 600 backend/cookies.txt && pm2 restart musicnote
  ```
  (playwright + chromium 필요: `.venv/bin/pip install playwright && .venv/bin/playwright install chromium`)
- 또는 브라우저 확장앱("Get cookies.txt LOCALLY", `youtube.com` 도메인만)으로 export.
- 쿠키는 **2~4주면 만료** → 주기적 재발급. `GET /api/health` 의 `yt_cookies` 로 인식 확인.
- `cookies.txt` 는 `.gitignore` 포함(민감정보), 권한 600.

### 2. JS "n" 챌린지 솔버 (yt-dlp-ejs + Deno)

YouTube는 스트림 URL을 주기 전에 JS `n` 서명 챌린지를 요구한다. yt-dlp 가 이걸 로컬에서 푼다:

- **`yt-dlp-ejs`** — venv 에 pip 설치됨 (`yt_dlp_ejs`). 챌린지 솔버 스크립트를 **오프라인**으로
  제공하므로 `--remote-components ejs:github` (런타임 원격 fetch) 가 필요 없다.
- **Deno** `~/.local/deno/bin/deno` — 솔버 실행 런타임. `ecosystem.config.js` 가 `musicnote` 의
  `PATH` 앞에 추가한다.
- ⚠ pm2 는 관리 프로세스에 `NODE_CHANNEL_FD`(IPC 소켓)를 넣는데, yt-dlp 가 띄우는 Deno 가
  이걸 상속받아 죽는다("fd is not from BiPipe"). `backend/fetch.py` 가 import 시점에 이 변수를
  제거한다 — 이 줄을 지우면 YouTube 다운로드가 "The page needs to be reloaded" 로 깨진다.

### 3. GVS PO 토큰 (bgutil provider)

- **bgutil PO-token provider** — pm2 앱 `bgutil-pot` (127.0.0.1:4416).
  Node ≥ 22 필요 → 시스템 node(20) 대신 `~/.local/node-v22.14.0-linux-arm64` 사용.
  소스: `~/bgutil-ytdlp-pot-provider` (tag 1.3.2). 재빌드:
  ```bash
  cd ~/bgutil-ytdlp-pot-provider/server
  ~/.local/node-v22.14.0-linux-arm64/bin/npm ci
  ~/.local/node-v22.14.0-linux-arm64/bin/npx tsc      # -> build/main.js
  pm2 restart bgutil-pot
  ```
  yt-dlp 플러그인: `bgutil-ytdlp-pot-provider==1.3.2` (venv 에 설치됨).
- `GET /api/health` 의 `pot_server` 로 동작 여부 확인.

### 디버깅

`fetch.py` 는 `MUSICNOTE_YTDL_DEBUG=1` 이면 yt-dlp verbose 로그를 `logs/err.log` 로 흘린다.
`player_client` 는 강제 지정하지 않는다(`tv`/`web_safari` 강제 시 SABR/UNPLAYABLE 로 깨짐).

쿠키가 없거나 만료돼도 **파일 업로드는 항상 동작**한다.

## 한계 (프로토타입)

- 최대 업로드 40 MB (`MUSICNOTE_MAX_MB` 로 조정, nginx `client_max_body_size` 도 함께).
- YouTube URL 은 유효한 `cookies.txt` + yt-dlp-ejs/Deno + bgutil PO provider 3종이 있어야 동작, 20분 이내만. 쿠키 만료(2~4주) 시 재발급 필요.
- 분석 job 은 백그라운드 스레드에서 동시 최대 4개(`app.MAX_JOBS`). 큰 파일은 수십 초~수 분.
- polyphonic 정확도는 basic-pitch 기준으로도 완벽하지 않음(옥타브/유령음 오류 가능).
- 결과(MIDI·재생용 오디오·작업 상태)는 30분 후 삭제.
