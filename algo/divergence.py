
import numpy as np

p = np.array([0.1, 0.3, 0.7])
q = np.array([0.2, 0.7, 0.1])
print('P =', p)
print('Q =', q)

# エントロピー
ent_p = -(p * np.log2(p)).sum()
print('ent(p) =', ent_p)
ent_q = -(q * np.log2(q)).sum()
print('ent(q) =', ent_q)

# KL情報量
kl_p_q = (p * (np.log2(p) - np.log2(q))).sum()
print('KL(P || Q) =', kl_p_q)

kl_q_p = (q * (np.log2(q) - np.log2(p))).sum()
print('KL(Q || P) =', kl_q_p)

r = np.array([0.45, 0.1, 0.45])
y = np.array([0.6, 0.2, 0.2])
z = np.array([0.35, 0.3, 0.35])

kl_r_y = (r * (np.log2(r) - np.log2(y))).sum()
kl_r_z = (r * (np.log2(r) - np.log2(z))).sum()

kl_y_r = (y * (np.log2(y) - np.log2(r))).sum()
kl_z_r = (z * (np.log2(z) - np.log2(r))).sum()

print(kl_r_y, kl_r_z)
print(kl_y_r, kl_z_r)
