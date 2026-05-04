import numpy as np
import os

# 1. 가상 데이터 생성 (단어 임베딩 32차원, 동작 시퀀스 90프레임)
word_embeddings = {"화장실": np.random.rand(32).astype(np.float32)}
gt_sequences = {"화장실": np.random.rand(90, 63).astype(np.float32)}

# 2. 저장 폴더 생성 및 저장
os.makedirs("dataset", exist_ok=True)
np.save("dataset/word_embeddings.npy", word_embeddings)
np.save("dataset/gt_sequences.npy", gt_sequences)

print("✅ [Prepare_data.py] 데이터셋 준비 완료")