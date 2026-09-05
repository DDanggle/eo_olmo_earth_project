"""cache_probes.py 단위 테스트. 합성 데이터로 각 프로브가 아는 답을 내는지 검증한다.

실행: env -u PYTHONPATH ./.venv-master/bin/python code/test_cache_probes.py
성공하면 exit 0, 하나라도 실패하면 exit 1 (체인이 여기서 멈추도록).
"""
import sys
import numpy as np

sys.path.insert(0, "/home/work/data/olmoearth/code")
from cache_probes import (physical_indices, block_pool, effective_rank,
                          participation_ratio, anisotropy, dead_channel_frac,
                          ridge_r2, fit_pca, B_RED, B_NIR, B_GREEN, B_SWIR1)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


print("== 밴드 계약 ==")
# B08(NIR)=3, B04(red)=2 인지 손계산으로 확인
raw = np.zeros((10, 3, 4, 4), dtype=np.uint16)
raw[B_NIR] = 3000     # NIR
raw[B_RED] = 1000     # red
raw[B_GREEN] = 500
raw[B_SWIR1] = 2000
p = physical_indices(raw)
check("NDVI = (NIR-RED)/(NIR+RED) = 0.5", np.allclose(p[0], 0.5, atol=1e-4), f"got {p[0,0,0]}")
check("NDWI = (GREEN-NIR)/(GREEN+NIR)", np.allclose(p[1], (500 - 3000) / 3500, atol=1e-4), f"got {p[1,0,0]}")
check("NDBI = (SWIR1-NIR)/(SWIR1+NIR)", np.allclose(p[2], (2000 - 3000) / 5000, atol=1e-4), f"got {p[2,0,0]}")
check("밴드 인덱스 NIR=3, RED=2", (B_NIR, B_RED) == (3, 2), f"got {(B_NIR, B_RED)}")

print("== 시간 집계는 중앙값 ==")
raw2 = np.zeros((10, 3, 2, 2), dtype=np.uint16)
raw2[B_NIR, 0] = 100; raw2[B_NIR, 1] = 200; raw2[B_NIR, 2] = 9000   # 이상치 1개
raw2[B_RED] = 100
p2 = physical_indices(raw2)
check("중앙값이 이상치를 무시", np.allclose(p2[0], (200 - 100) / 300, atol=1e-3), f"got {p2[0,0,0]}")

print("== block_pool ==")
a = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
b = block_pool(a, 4, 4)
check("모양 (2,4,4)", b.shape == (2, 4, 4), str(b.shape))
check("좌상단 = 2x2 평균", np.isclose(b[0, 0, 0], a[0, :2, :2].mean()), f"{b[0,0,0]} vs {a[0,:2,:2].mean()}")

print("== effective_rank / participation_ratio ==")
rng = np.random.default_rng(0)
iso = rng.normal(size=(4000, 32))
check("등방 가우시안의 유효rank가 차원에 근접", effective_rank(iso) > 0.85 * 32, f"{effective_rank(iso):.1f}")
u = rng.normal(size=(4000, 1)); r1 = u @ rng.normal(size=(1, 32))
check("rank-1의 유효rank ~ 1", effective_rank(r1) < 1.5, f"{effective_rank(r1):.3f}")
check("rank-1의 PR ~ 1", participation_ratio(r1) < 1.5, f"{participation_ratio(r1):.3f}")
check("등방 PR이 차원에 근접", participation_ratio(iso) > 0.85 * 32, f"{participation_ratio(iso):.1f}")

print("== anisotropy ==")
check("등방 가우시안 ~ 0", abs(anisotropy(iso, rng)) < 0.05, f"{anisotropy(iso, rng):.3f}")
one = np.ones((2000, 16)) + 0.001 * rng.normal(size=(2000, 16))
check("거의 상수 ~ 1", anisotropy(one, rng) > 0.95, f"{anisotropy(one, rng):.3f}")

print("== dead_channel_frac ==")
z = rng.normal(size=(1000, 10)); z[:, :3] = 0.0
check("죽은 채널 3/10", abs(dead_channel_frac(z) - 0.3) < 1e-9, f"{dead_channel_frac(z)}")

print("== ridge_r2 ==")
X = rng.normal(size=(2000, 20)); w = rng.normal(size=(20, 2))
Y = X @ w
Xte = rng.normal(size=(500, 20)); Yte = Xte @ w
r2 = ridge_r2(X, Y, Xte, Yte)
check("완전 선형 관계면 R^2 ~ 1", all(v > 0.98 for v in r2), str(r2))
Yr = rng.normal(size=(2000, 2)); Yter = rng.normal(size=(500, 2))
r2n = ridge_r2(X, Yr, Xte, Yter)
check("무관계면 R^2 <= 0.1", all(v < 0.1 for v in r2n), str(r2n))
# 기준선이 '학습 평균'인지 '테스트 평균'인지를 실제로 구분하는 테스트.
# 정보 없는 X + 테스트 목표를 크게 이동:
#   학습평균 기준이면  ss_res ~ ss_tot ~ n*shift^2  ->  R^2 ~ 0
#   테스트평균 기준이면 ss_tot ~ n*var(작음)        ->  R^2 << 0 (크게 음수)
Xu = rng.normal(size=(2000, 20)); Xute = rng.normal(size=(500, 20))
Yu = rng.normal(size=(2000, 1)); Yute = rng.normal(size=(500, 1)) + 50.0
r2_shift = ridge_r2(Xu, Yu, Xute, Yute)[0]
check("학습평균 기준: 이동해도 R^2 ~ 0 (음수 폭발 아님)",
      abs(r2_shift) < 0.2, f"{r2_shift:.4f}")
# 같은 상황을 테스트평균 기준으로 손계산하면 크게 음수여야 한다 (대조)
mx, my = Xu.mean(0), Yu.mean(0)
Xc, Yc = Xu - mx, Yu - my
lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
P = (Xute - mx) @ W + my
r2_testmean = float(1 - ((Yute - P) ** 2).sum() / (((Yute - Yute.mean(0)) ** 2).sum() + 1e-12))
check("대조: 테스트평균 기준이면 크게 음수", r2_testmean < -100, f"{r2_testmean:.1f}")
check("두 기준이 실제로 다르다", abs(r2_shift - r2_testmean) > 100,
      f"train-mean {r2_shift:.3f} vs test-mean {r2_testmean:.1f}")

print("== PCA ==")
Xp = rng.normal(size=(1000, 40)) @ rng.normal(size=(40, 40))
mu, V = fit_pca(Xp, 8)
check("PCA 모양 (40,8)", V.shape == (40, 8), str(V.shape))
check("PCA 축 직교", np.allclose(V.T @ V, np.eye(8), atol=1e-6))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
