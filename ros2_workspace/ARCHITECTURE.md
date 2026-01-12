# Agent McQueen - Architecture

## Overview

강화학습으로 학습된 자율주행 에이전트를 F1Tenth/ForzaETH 시뮬레이터와 연동하는 ROS2 패키지

### 에이전트 종류

| Agent | 설명 | 모델 |
|-------|------|------|
| **Stage 1** | 단일 에이전트 트랙 주행 | stable_baselines3 PPO |
| **Overtake** | 두 에이전트 추월 시뮬레이션 | Custom OvertakePolicy |

### Overtake 구성

- **Agent 0 (ego)**: Stage 1 frozen expert, 80% 속도, 앞에서 시작
- **Agent 1 (opp)**: Overtake-trained, 100% 속도, 뒤에서 시작하여 추월 시도

---

## Directory Structure

```
AgentMcqueen_ws/
└── src/agent_mcqueen/
    ├── agent_mcqueen/
    │   ├── stage1_agent_node.py        # Stage 1 ROS2 노드
    │   └── overtake_agent_node.py      # Overtake ROS2 노드
    ├── config/
    │   ├── sim_stage1.yaml             # F1Tenth Stage 1 시뮬레이터 설정
    │   ├── stage1_config.yaml          # F1Tenth Stage 1 에이전트 파라미터
    │   ├── sim_overtake.yaml           # F1Tenth Overtake 시뮬레이터 설정
    │   ├── overtake_config.yaml        # F1Tenth Overtake 에이전트 파라미터
    │   ├── forza_stage1_config.yaml    # ForzaETH Stage 1 에이전트 파라미터
    │   └── forza_overtake_config.yaml  # ForzaETH Overtake 에이전트 파라미터
    └── launch/
        ├── sim_stage1_launch.py        # F1Tenth + Stage 1 런치
        ├── sim_overtake_launch.py      # F1Tenth + Overtake 런치
        ├── forza_stage1_launch.py      # ForzaETH + Stage 1 런치
        └── forza_overtake_launch.py    # ForzaETH + Overtake 런치
```

---

## Prerequisites

### Docker 환경

ForzaETH Docker 컨테이너 내부에서 실행:

```bash
# 호스트에서
cd /path/to/forzaeth
./run_docker.sh
```

### 필수 워크스페이스

| 경로 | 설명 |
|------|------|
| `/home/misys/f1tenth_ws` | F1Tenth Gym ROS2 브릿지 |
| `/home/misys/forza_ws/race_stack` | ForzaETH Race Stack |
| `/home/misys/AgentMcqueen_ws` | 이 패키지 |
| `/home/misys/overtake_agent` | 학습된 모델 및 공통 라이브러리 |

### 필수 모델 파일

| 모델 | 경로 |
|------|------|
| Stage 1 | `/home/misys/overtake_agent/common/models/stage1/.../final.zip` |
| Overtake | `/home/misys/overtake_agent/common/models/overtake_final.pth` |

---

## Technical Details

### Observation 처리

#### LiDAR Scan
```python
scan = np.clip(scan, 0.0, 10.0) / 10.0  # [0, 1] 정규화
scan = scan[::-1]                        # 방향 반전 (F110 convention)
```

#### Velocity
```python
velocity = vx / 3.2  # 최대 속도로 정규화
```

#### Frame Stacking
- 4개 프레임 스택 → shape: (4, 1081)
- 각 프레임: 1080 LiDAR beams + 1 velocity

### Observation 순서 (Stage 1)

`FlattenObservation`은 Dict를 알파벳 순으로 정렬:
```python
# 올바른 순서: [linear_vel, scans] (알파벳순)
obs = np.concatenate([velocity, scan])  # NOT [scan, velocity]
```

### Overtake Asymmetric Observations

| Agent | Dimension | 구성 |
|-------|-----------|------|
| Agent 0 | 4324 | scan(1080) + vel(1) × 4 frames |
| Agent 1 | 4336 | scan(1080) + vel(1) + delta_s(1) + delta_vs(1) + ahead(1) × 4 frames |

---

## ROS2 Topics

### F1Tenth Stage 1

| Topic | Type | Direction |
|-------|------|-----------|
| `/sim/scan` | LaserScan | Subscribe |
| `/sim/ego_racecar/odom` | Odometry | Subscribe |
| `/sim/drive` | AckermannDriveStamped | Publish |

### F1Tenth Overtake

| Topic | Type | Direction | Agent |
|-------|------|-----------|-------|
| `/sim/scan` | LaserScan | Subscribe | Ego |
| `/sim/opp_scan` | LaserScan | Subscribe | Opp |
| `/sim/ego_racecar/odom` | Odometry | Subscribe | Ego |
| `/sim/opp_racecar/odom` | Odometry | Subscribe | Opp |
| `/sim/drive` | AckermannDriveStamped | Publish | Ego |
| `/sim/opp_drive` | AckermannDriveStamped | Publish | Opp |

### ForzaETH Stage 1

| Topic | Type | Direction |
|-------|------|-----------|
| `/scan` | LaserScan | Subscribe |
| `/car_state/odom_GT` | Odometry | Subscribe |
| `/drive` | AckermannDriveStamped | Publish |

### ForzaETH Overtake

| Topic | Type | Direction | Agent |
|-------|------|-----------|-------|
| `/scan` | LaserScan | Subscribe | Ego |
| `/opp_scan` | LaserScan | Subscribe | Opp |
| `/car_state/odom_GT` | Odometry | Subscribe | Ego |
| `/opp_racecar/odom` | Odometry | Subscribe | Opp |
| `/drive` | AckermannDriveStamped | Publish | Ego |
| `/opp_drive` | AckermannDriveStamped | Publish | Opp |

---

## Configuration Reference

### sim_overtake.yaml

```yaml
# 에이전트 시작 위치
sx: 0.2663860       # Agent 0 (앞)
sx1: 10.2663860     # Agent 1 (뒤, 약 10m 차이)
stheta: 3.208718    # ≈ π, -x 방향 → 낮은 x가 앞

# 맵 설정
map_path: '/home/misys/overtake_agent/f1tenth/maps/map0'
num_agent: 2
```

### overtake_config.yaml

```yaml
# 속도 핸디캡
agent0_speed_factor: 0.8  # Agent 0은 80% 속도

# 모델 경로
model_path: '/home/misys/overtake_agent/common/models/overtake_final.pth'

# 디바이스
device: 'cuda'  # GPU 없으면 자동으로 'cpu'
```

---

## Integration Status

| 시뮬레이터 | Stage 1 | Overtake | 상태 |
|-----------|---------|----------|------|
| F1Tenth Gym | O | O | 완료 |
| ForzaETH | O | O | 완료 |

### ForzaETH 사용 가능한 맵

| 맵 이름 | 설명 |
|--------|------|
| `hall` | 넓은 복도 환경 |
| `GLC_smile_small` | 소형 트랙 |
| `small_hall` | 소형 홀 |
| `teras` | 테라스/복도 환경 |
| `glc_ot_ez` | 추월 훈련용 트랙 |
| `hangar_1905_v0` | 격납고 환경 |
