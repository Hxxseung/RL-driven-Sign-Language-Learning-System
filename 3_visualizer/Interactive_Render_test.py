"""
Interactive_Render_test.py  ―  GRAPS 시스템 최종 통합 버전 (Qwen DPO + PPO Blender)
"""

import os
import sys
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from stable_baselines3 import PPO
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 💡 2_motion_blender 폴더를 모듈 검색 경로에 추가 ──
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_BLENDER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '2_motion_blender'))
sys.path.append(MOTION_BLENDER_DIR)

from SignCorrectionEnv import SignCorrectionEnv, load_sign_dictionary


# ── 한글 폰트 설정 ──────────────────────────────────────────────
_KO_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_ko_font_prop = None
for _p in _KO_FONT_CANDIDATES:
    if os.path.exists(_p):
        _ko_font_prop = fm.FontProperties(fname=_p)
        fm.fontManager.addfont(_p)
        plt.rcParams['font.family'] = fm.FontProperties(fname=_p).get_name()
        break
if _ko_font_prop is None:
    _ko_font_prop = fm.FontProperties()


# ── 스켈레톤 토폴로지 ──────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
]
PANEL_BG = ['#EBF5FF', '#FFF3E0', '#F0FFF4', '#FFF0F5']


# ── 헬퍼 함수 ────────────────────────────────────────────────────
def _axis_limits(seq: np.ndarray, margin_ratio: float = 0.08):
    pts   = seq.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    x_pad = max((x_max - x_min) * margin_ratio, 1.0)
    y_pad = max((y_max - y_min) * margin_ratio, 1.0)
    return (x_min - x_pad, x_max + x_pad), (-y_max - y_pad, -y_min + y_pad)


def _draw_frame(ax, frame_data, title="", bg_color='white', xlim=None, ylim=None):
    ax.cla()
    ax.set_facecolor(bg_color)
    pts     = frame_data.reshape(-1, 2)
    left_h  = pts[:21]
    right_h = pts[21:42]
    pose    = pts[42:]

    def _part(points, conns, color):
        ax.scatter(points[:, 0], -points[:, 1], c=color, s=18, zorder=3)
        for i, j in conns:
            if i < len(points) and j < len(points):
                ax.plot(
                    [points[i,0], points[j,0]],
                    [-points[i,1], -points[j,1]],
                    c=color, lw=1.1, alpha=0.75
                )

    _part(left_h,  HAND_CONNECTIONS, '#4A90D9')
    _part(right_h, HAND_CONNECTIONS, '#E05252')
    _part(pose,    POSE_CONNECTIONS, '#4CAF50')

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=8, fontproperties=_ko_font_prop)
    ax.axis('off')
    ax.set_aspect('equal')


def _normalize_len(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) >= target_len:
        return seq[:target_len].astype(np.float32)
    pad = np.tile(seq[-1:], (target_len - len(seq), 1))
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


def _parse_glosses_from_response(response: str) -> list:
    """
    Qwen 출력에서 글로스 리스트를 추출합니다.
    오직 'GLOSSES:' 라인에 있는 단어만 엄격하게 추출하며 줄글 파싱을 원천 차단합니다.
    """
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("GLOSSES:"):
            parts = line.split(":", 1)[1].strip().split()
            
            cleaned_parts = []
            for p in parts:
                clean_word = re.sub(r'[^\w가-힣]', '', p)
                # 🔥 수정: 1글자 단어(별, 달, 해 등)도 통과하도록 >= 1 로 변경
                if len(clean_word) >= 1 and clean_word not in ['수어', '단어', '의미', '표현']:
                    cleaned_parts.append(clean_word)
            return cleaned_parts
            
    return []


class FullPipelineSignGenerator:

    def __init__(self,
                 qwen_model_path:      str = "../1_llm_decomposer/best_qwen_sign_decomposer",
                 selector_model_path:  str = "../2_motion_blender/sign_selector_model",
                 data_dir:             str = "../dataset_processed",
                 dict_path:            str = "../sign_dict.txt",
                 dict_xlsx:            str = "../sign_dict_vocab.xlsx",
                 total_frames:         int = 90):

        print("📦 1단계: Qwen DPO 분해 에이전트 로드 중...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.qwen_tokenizer.padding_side = "left"
        self.qwen_tokenizer.pad_token    = self.qwen_tokenizer.eos_token

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.qwen_model.eval()

        print("📦 2단계: 글로스 선택 에이전트 로드 중...")
        self.selector_env   = SignCorrectionEnv(data_dir=data_dir, dict_xlsx=dict_xlsx)
        self.selector_model = PPO.load(selector_model_path, device="cpu")
        self.total_frames   = total_frames

        if os.path.exists(dict_xlsx):
            self.word_to_id, self.id_to_words = load_sign_dictionary(dict_xlsx)
        else:
            self.word_to_id, self.id_to_words = {}, {}

        gt_data  = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))
        self.gt_sequences    = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys        = list(self.gt_sequences.keys())

        self.registered_glosses = set(k.split('_')[0] for k in self.all_keys)
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        self.registered_glosses.add(w)

        print(f"✅ 시스템 준비 완료! (사전 등록 단어 수: {len(self.registered_glosses):,}개)")

    def _find_similar_glosses(self, query: str, top_k: int = 5) -> list:
        result = []
        seen   = set()

        if query in self.word_to_id:
            tid      = self.word_to_id[query]
            synonyms = self.id_to_words.get(tid, [])
            for s in synonyms:
                if s != query and s not in seen and s in self.registered_glosses:
                    result.append(s)
                    seen.add(s)

        if len(result) < top_k and len(query) >= 2:
            for w in self.word_to_id:
                if w in seen or w == query:
                    continue
                if len(w) >= 2 and (query in w or w in query) and w in self.registered_glosses:
                    result.append(w)
                    seen.add(w)
                if len(result) >= top_k:
                    break

        return result[:top_k]

    def _call_qwen(self, chat_prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.qwen_tokenizer(
            chat_prompt, return_tensors="pt"
        ).to(self.qwen_model.device)
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _decompose_with_qwen(self, word: str, max_retries: int = 5) -> list:
        system_msg = (
            "너는 입력된 복합어를 우리가 보유한 수어 사전 단어 리스트에 맞추어 분해하는 수어 통역 에이전트야.\n"
            "분해 이유를 설명하되, 맨 마지막 줄에는 반드시 'GLOSSES: 단어1 단어2' 형식으로 최종 결과만 써야 해.\n"
            "⚠️ 주의: GLOSSES 라인에는 '수어', '단어', '의미' 같은 설명용 단어를 절대 포함하지 마."
        )

        first_user_msg = (
            f"다음 복합어를 수어 사전에 등록된 단어 단위로 분해하세요.\n"
            f"입력 단어: {word}"
        )
        # 🔥 수정: 히스토리 누적 방지 (초기 상태 설정)
        history = [{"role": "user", "content": first_user_msg}]

        for attempt in range(1, max_retries + 1):
            chat_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            for turn in history:
                role = "user" if turn["role"] == "user" else "assistant"
                chat_prompt += f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n"
            chat_prompt += "<|im_start|>assistant\n"

            response = self._call_qwen(chat_prompt, max_new_tokens=256)
            
            print(f"\n==== [ Qwen 분해 시도 {attempt}/{max_retries} ] ====")
            print(response)
            print("=" * 44 + "\n")

            predicted_glosses = _parse_glosses_from_response(response)
            
            if not predicted_glosses:
                print(f"[파싱 실패]: GLOSSES 포맷을 찾을 수 없습니다.")
                feedback = (
                    f"이전 답변에서 'GLOSSES: 단어1 단어2' 형식을 찾을 수 없습니다.\n"
                    f"다시 생각하되, 반드시 마지막 줄에 'GLOSSES: '로 시작하여 분해된 단어만 나열하세요."
                )
                print(f"  ❌ 포맷 오류 → 포맷 재강조 피드백 후 재시도 ({attempt}/{max_retries})")
                # 🔥 토큰 에러 방지: 과거 실패 기록 지우고 새 피드백으로만 덮어쓰기
                history = [
                    {"role": "user", "content": first_user_msg},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback}
                ]
                continue

            print(f"[파싱된 글로스]: {predicted_glosses}")

            valid_parts = []
            failed_glosses = []
            for gloss in predicted_glosses:
                matched_keys = [k for k in self.all_keys if k.split('_')[0] == gloss]
                if matched_keys:
                    valid_parts.append((matched_keys[0], gloss))
                else:
                    failed_glosses.append(gloss)

            if valid_parts and not failed_glosses:
                return valid_parts
            
            feedback_lines = []
            for g in failed_glosses:
                similar = self._find_similar_glosses(g, top_k=5)
                if similar:
                    sim_str = ", ".join(f"[{s}]" for s in similar)
                    feedback_lines.append(f"  - [{g}]: 데이터셋에 없음. 대체 추천 후보 → {sim_str}")
                else:
                    feedback_lines.append(f"  - [{g}]: 데이터셋에 없음. 의미가 통하는 다른 기초 수어 단어들로 돌려서 다시 설명할 것.")
            
            feedback = (
                f"방금 제시한 글로스 중 데이터셋에 없는 단어가 있습니다:\n"
                + "\n".join(feedback_lines)
                + f"\n\n추천 후보가 있다면 그 안에서 고르고, 없다면 [{word}]의 본래 의미를 잘 나타낼 수 있는 완전히 다른 기초 단어들의 조합으로 다시 분해해 주세요.\n"
                f"⚠️ 주의: 이유를 설명한 뒤, 맨 마지막 줄에는 반드시 이전처럼 'GLOSSES: 단어1 단어2' 형식으로 최종 분해된 단어만 작성해야 합니다."
            )
            print(f"  ❌ 미매칭 단어 발생 → 유사어 피드백 후 재시도 ({attempt}/{max_retries})")
            
            # 🔥 토큰 에러 방지: 과거 대화 지우고 최신 피드백만 남겨서 길이 최적화
            history = [
                {"role": "user", "content": first_user_msg},
                {"role": "assistant", "content": response},
                {"role": "user", "content": feedback}
            ]

        print(f"  ❌ {max_retries}회 재시도 모두 실패: [{word}]")
        return []

    # ── 🏃 수어 생성 메인 파이프라인 ──────────────────────────────
    def generate(self, word: str) -> dict:
        exact_keys = [k for k in self.all_keys if k.split('_')[0] == word]
        if exact_keys:
            key    = exact_keys[0]
            gt_seq = _normalize_len(self.gt_sequences[key], self.total_frames)
            print(f"✅ 사전 직접 등재 단어: '{word}' (즉시 표출)")
            return {
                "word":        word,
                "output_seq":  gt_seq,
                "output_type": "gt",
                "gt_parts":    [{"gt_seq": gt_seq, "gloss": word}],
            }

        parts = self._decompose_with_qwen(word)

        if not parts:
            raise ValueError(f"'{word}'의 분해 결과를 모두 데이터셋에서 찾을 수 없어 렌더링을 중단합니다.")

        if len(parts) == 1:
            key, gloss = parts[0]
            ref_gt = _normalize_len(self.gt_sequences[key], self.total_frames)
            return {
                "word":        word,
                "output_seq":  ref_gt,
                "output_type": "gt",
                "gt_parts":    [{"gt_seq": ref_gt, "gloss": gloss}],
            }

        glosses = [p[1] for p in parts]
        candidates = {}
        for key, gloss in parts:
            all_cands = [k for k in self.all_keys if k.split('_')[0] == gloss]
            candidates[gloss] = all_cands[:self.selector_env.MAX_CANDIDATES]

        selected = self.selector_env.select_keys(glosses, candidates, self.selector_model)
        print(f"🎯 PPO 키 선택 결과:")
        for gloss, chosen_key in selected.items():
            gid = self.word_to_id.get(gloss, "미등록")
            print(f"   [{gloss}] → {chosen_key}  (표제어 그룹: {gid})")

        gt_parts = []
        seqs     = []
        for gloss in glosses:
            key = selected.get(gloss)
            if key and key in self.gt_sequences:
                seq = _normalize_len(self.gt_sequences[key], self.total_frames)
                gt_parts.append({"gt_seq": seq, "gloss": gloss})
                seqs.append(seq)

        if not seqs:
            raise ValueError(f"선택된 키에서 시퀀스를 로드할 수 없습니다.")

        output_seq = np.concatenate(seqs, axis=0)
        self.current_total_frames = len(output_seq)

        return {
            "word":        word,
            "output_seq":  output_seq,
            "output_type": "sequential",
            "gt_parts":    gt_parts,
        }

    # ── 💾 GIF 저장 ───────────────────────────────────────────────
    def save_gif(self, result: dict, out_path: str, fps: int = 15):
        word        = result["word"]
        output_seq  = result["output_seq"]
        output_type = result["output_type"]
        gt_parts    = result["gt_parts"]
        n_gt        = len(gt_parts)

        n_cols = 1 + n_gt
        fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 6))
        axes = [axes] if n_cols == 1 else list(axes)

        title_str = (
            f"입력: {word}  (Qwen 분해 + PPO 키 선택 → 순차 연결)"
            if output_type == "sequential"
            else f"입력: {word}  (사전 직접 표출)"
        )
        fig.suptitle(
            title_str, fontsize=12, fontweight='bold',
            fontproperties=_ko_font_prop, y=1.01
        )

        n_frames = len(output_seq)
        ppo_xlim, ppo_ylim = _axis_limits(output_seq)
        gt_xlim,  gt_ylim  = _axis_limits(
            np.concatenate([gp["gt_seq"] for gp in gt_parts], axis=0)
        )
        left_label = "GT 직접 표출" if output_type == "gt" else "순차 연결 결과"

        def _update(f):
            _draw_frame(
                axes[0], output_seq[f],
                title=f"{left_label} ({f+1}/{n_frames}f)",
                bg_color='white', xlim=ppo_xlim, ylim=ppo_ylim
            )
            for i, gp in enumerate(gt_parts):
                gt_f = min(f, len(gp["gt_seq"]) - 1)
                bg   = PANEL_BG[i % len(PANEL_BG)] if n_gt > 1 else '#F5F5F5'
                _draw_frame(
                    axes[i + 1], gp["gt_seq"][gt_f],
                    title=f"구성 요소 [{gp['gloss']}]",
                    bg_color=bg, xlim=gt_xlim, ylim=gt_ylim
                )
            return []

        ani = animation.FuncAnimation(
            fig, _update, frames=n_frames,
            interval=1000 // fps, blit=False
        )
        ani.save(out_path, writer='pillow', fps=fps)
        plt.close()
        print(f"🎉 GIF 저장 완료: {out_path}\n")


# ── 메인 구동 루프 ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        pipeline = FullPipelineSignGenerator(
            qwen_model_path     ="../1_llm_decomposer/best_qwen_sign_decomposer",
            selector_model_path ="../2_motion_blender/sign_selector_model",
            data_dir            ="../dataset_processed",
            dict_path           ="../sign_dict.txt",
            dict_xlsx           ="../sign_dict_vocab.xlsx",
        )
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        exit(1)

    print("\n🖥️  [GRAPS End-to-End 통합 모니터 기동]")
    print("   Qwen이 복합어를 분해하고, PPO가 표제어 사전 기반으로 최적 키를 선택합니다.\n")

    while True:
        query = input("👉 단어 입력 (q: 종료): ").strip()
        if query.lower() in ('q', 'exit', '종료'):
            break
        if not query:
            continue
        try:
            res = pipeline.generate(query)
            pipeline.save_gif(res, f"result_{query}.gif", fps=15)
        except ValueError as e:
            print(f"⚠️  {e}")
        except Exception as e:
            print(f"🔥 렌더링 오류: {e}")