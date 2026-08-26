from __future__ import annotations

"""
LLM Provider Layer
==================
Một chỗ duy nhất quyết định gọi LLM nào cho: answer generation (pipeline),
enrichment (M5) và RAGAS judge (M4).

Thứ tự ưu tiên: GOOGLE_API_KEY (Gemini) → OPENAI_API_KEY (OpenAI) → None (fallback
extractive, pipeline vẫn chạy end-to-end nhưng RAGAS = 0).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ACTIVE_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_CHAT_MODEL,
                    GEMINI_JUDGE_MODEL,
                    GEMINI_EMBED_MODEL, OPENAI_CHAT_MODEL, LLM_MAX_RETRIES, LLM_RPM,
                    LLM_TIMEOUT)

_GEMINI_MODELS: dict[str, object] = {}
_OPENAI_CLIENT = None


def provider() -> str:
    """Trả về provider đang dùng: "gemini" | "openai" | "none".

    Do config.ACTIVE_PROVIDER quyết định, để LLM_PROVIDER trong .env chọn được
    tường minh khi máy có sẵn cả hai key.
    """
    return ACTIVE_PROVIDER


def has_llm() -> bool:
    return provider() != "none"


# ─── Gemini ──────────────────────────────────────────────


# Gemini 3.x bật "thinking" mặc định: thinking tokens tính chung vào max_output_tokens,
# nên budget nhỏ (100-400) sẽ bị thinking ăn hết và trả về text rỗng/cụt.
# Đặt sàn đủ rộng để phần trả lời thật luôn được sinh ra.
GEMINI_MIN_OUTPUT_TOKENS = 2048


def _gemini_model(json_mode: bool, max_tokens: int, system: str, model: str | None = None):
    """Cache model theo (model, json_mode, max_tokens, system) — tránh khởi tạo lại mỗi call."""
    import google.generativeai as genai

    model_name = model or GEMINI_CHAT_MODEL
    max_tokens = max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS)
    key = f"{model_name}|{json_mode}|{max_tokens}|{system}"
    if key not in _GEMINI_MODELS:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {"temperature": 0.0, "max_output_tokens": max_tokens}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        _GEMINI_MODELS[key] = genai.GenerativeModel(
            model_name,
            system_instruction=system or None,
            generation_config=generation_config,
        )
    return _GEMINI_MODELS[key]


# ─── Public API ──────────────────────────────────────────


# Cửa sổ trượt tách riêng theo model: quota free tier của Gemini là
# "GenerateRequestsPerMinutePerProjectPerModel" — mỗi model id có hạn mức riêng,
# nên đếm gộp sẽ throttle chặt hơn mức cần thiết.
_CALL_TIMES_BY_MODEL: dict[str, list[float]] = {}
# Tổng thời gian đã ngủ vì rate limit (giây). Phase C trừ phần này ra khỏi số đo
# latency của guardrail — chờ quota không phải là latency của rail.
_THROTTLE_WAIT_TOTAL = 0.0


def throttle_wait_total() -> float:
    """Tổng số giây đã chờ vì rate limiter, tính từ khi process bắt đầu."""
    return _THROTTLE_WAIT_TOTAL


def _throttle(model: str | None = None) -> None:
    """Rate limiter cửa sổ trượt: tối đa LLM_RPM request trong 60 giây.

    Free tier Gemini = 15 request/phút. Nếu gọi liên tục, request thứ 16 bị 429 và
    rơi vào fallback `answer = contexts[0]` — làm faithfulness bị thổi lên giả
    (answer trùng context nên luôn "grounded") và answer_relevancy tụt.

    Dùng cửa sổ trượt thay vì sleep cố định giữa 2 call: những call lẻ (test, tra cứu
    đơn) không phải chờ, chỉ khi chạm hạn mức mới chờ đúng phần cần thiết.
    """
    global _THROTTLE_WAIT_TOTAL

    if LLM_RPM <= 0:
        return

    window = _CALL_TIMES_BY_MODEL.setdefault(model or GEMINI_CHAT_MODEL, [])
    now = time.time()
    window[:] = [t for t in window if now - t < 60.0]
    if len(window) >= LLM_RPM:
        wait = 60.0 - (now - window[0]) + 0.5
        if wait > 0:
            print(f"  ⏳ Đạt {LLM_RPM} request/phút ({model or GEMINI_CHAT_MODEL}) "
                  f"— chờ {wait:.0f}s", flush=True)
            time.sleep(wait)
            _THROTTLE_WAIT_TOTAL += wait
        now = time.time()
        window[:] = [t for t in window if now - t < 60.0]
    window.append(now)


# Lock phải tạo riêng cho từng event loop: RAGAS gọi evaluate() một lần mỗi lô và
# mỗi lần lại dựng event loop mới, dùng lại Lock cũ sẽ vỡ với
#   RuntimeError: <asyncio.locks.Lock> is bound to a different event loop
_ATHROTTLE_LOCKS: dict[int, "object"] = {}


async def _athrottle(model: str | None = None) -> None:
    """Bản async của _throttle(), dùng chung cửa sổ trượt _CALL_TIMES.

    RAGAS gọi judge LLM qua đường async với nhiều worker song song. Nếu dùng
    _throttle() (time.sleep) ở đây thì cả event loop bị block; còn nếu không
    throttle thì free tier 15 req/phút trả 429 hàng loạt, langchain retry cạn
    lượt và metric rơi về NaN → điểm bị tính thành 0.0.
    """
    global _THROTTLE_WAIT_TOTAL
    import asyncio

    if LLM_RPM <= 0:
        return

    loop_key = id(asyncio.get_running_loop())
    lock = _ATHROTTLE_LOCKS.get(loop_key)
    if lock is None:
        lock = _ATHROTTLE_LOCKS[loop_key] = asyncio.Lock()
        # Loop cũ đã đóng — giữ lại lock của chúng chỉ tổ rò rỉ bộ nhớ.
        for stale in [k for k in _ATHROTTLE_LOCKS if k != loop_key]:
            _ATHROTTLE_LOCKS.pop(stale, None)

    async with lock:
        window = _CALL_TIMES_BY_MODEL.setdefault(model or GEMINI_JUDGE_MODEL, [])
        now = time.time()
        window[:] = [t for t in window if now - t < 60.0]
        if len(window) >= LLM_RPM:
            wait = 60.0 - (now - window[0]) + 0.5
            if wait > 0:
                await asyncio.sleep(wait)
                _THROTTLE_WAIT_TOTAL += wait
            now = time.time()
            window[:] = [t for t in window if now - t < 60.0]
        window.append(now)


def _retry_delay_from_error(err: Exception, fallback: float) -> float:
    """Đọc `retry_delay { seconds: N }` mà Gemini trả về trong lỗi 429."""
    import re
    match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(err))
    if match:
        return float(match.group(1)) + 1.0
    return fallback


def chat(system: str, user: str, json_mode: bool = False, max_tokens: int = 512,
         model: str | None = None) -> str | None:
    """Gọi LLM 1 lượt. Trả về text, hoặc None nếu không có provider / lỗi hết retry.

    `model` cho phép override model mặc định (Phase B dùng JUDGE_MODEL khác generator
    để tránh self-preference bias).

    Có throttle + retry theo `retry_delay` mà API trả về, vì Gemini free tier giới
    hạn 15 request/phút (429 RESOURCE_EXHAUSTED).
    """
    p = provider()
    if p == "none":
        return None

    last_error = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            _throttle()
            if p == "gemini":
                gemini = _gemini_model(json_mode, max_tokens, system, model)
                # request_options.timeout: không có nó, một socket treo sẽ block
                # vĩnh viễn (đã gặp: pipeline đứng 10 phút không log gì).
                resp = gemini.generate_content(
                    user, request_options={"timeout": LLM_TIMEOUT})
                return (resp.text or "").strip()

            global _OPENAI_CLIENT
            if _OPENAI_CLIENT is None:
                from openai import OpenAI
                _OPENAI_CLIENT = OpenAI()
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = _OPENAI_CLIENT.chat.completions.create(
                timeout=LLM_TIMEOUT,
                model=model or OPENAI_CHAT_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            # Hết quota NGÀY thì retry vô nghĩa (phải chờ tới hôm sau) -> fallback ngay
            if "PerDay" in str(e):
                print(f"  ⚠️  Hết quota ngày của {GEMINI_CHAT_MODEL} — dùng fallback extractive.")
                return None
            if attempt < LLM_MAX_RETRIES - 1:
                # 429 kèm retry_delay -> chờ đúng khoảng API yêu cầu, còn lại backoff 2/4/8s
                delay = _retry_delay_from_error(e, 2 ** attempt * 2)
                # Log mỗi lần retry: im lặng khi retry khiến pipeline trông như treo.
                print(f"  ⚠️  LLM lỗi ({type(e).__name__}), thử lại sau {delay:.0f}s "
                      f"[{attempt + 1}/{LLM_MAX_RETRIES}]", flush=True)
                time.sleep(delay)

    print(f"  ⚠️  LLM call failed sau {LLM_MAX_RETRIES} lần thử: {last_error}")
    return None


def _ragas_compat_gemini():
    """ChatGoogleGenerativeAI vá lỗi tương thích với RAGAS 0.1.x.

    RAGAS gọi `generate_prompt(..., temperature=..., n=...)`, các kwarg này được
    truyền thẳng xuống google client và gây:
        TypeError: generate_content() got an unexpected keyword argument 'temperature'
    Ở đây ta chuyển temperature vào generation_config (đúng chỗ của nó) và bỏ `n`
    (RAGAS đã tự nhân bản prompt khi cần nhiều completion).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    class RagasCompatGemini(ChatGoogleGenerativeAI):
        @staticmethod
        def _fix_kwargs(kwargs: dict) -> dict:
            temperature = kwargs.pop("temperature", None)
            kwargs.pop("n", None)
            gen_config = dict(kwargs.pop("generation_config", None) or {})
            if temperature is not None:
                gen_config["temperature"] = float(temperature)
            gen_config.setdefault("max_output_tokens", GEMINI_MIN_OUTPUT_TOKENS)
            kwargs["generation_config"] = gen_config
            return kwargs

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            _throttle(self.model)
            return super()._generate(messages, stop=stop, run_manager=run_manager,
                                     **self._fix_kwargs(kwargs))

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            await _athrottle(self.model)
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager,
                                            **self._fix_kwargs(kwargs))

    return RagasCompatGemini


def ragas_backend(judge_model: str | None = None):
    """Trả về (llm, embeddings) cho RAGAS, hoặc (None, None) để RAGAS dùng default OpenAI.

    RAGAS 0.1.x nhận trực tiếp LangChain LLM/Embeddings và tự wrap.
    Dùng embeddings của Gemini qua API → không tốn RAM cho model local.

    judge_model: ghi đè model judge. Cho phép mỗi RAGAS metric dùng một model riêng
    để chia tải rate limit (xem RAGAS_JUDGE_POOL trong src/m4_eval.py).
    """
    p = provider()
    if p == "openai":
        # RAGAS 0.1.x mặc định đã dùng OpenAI: trả (None, None) để nó tự dựng
        # ChatOpenAI + OpenAIEmbeddings, khỏi cần lớp vá tương thích nào.
        # RAGAS đọc key từ biến môi trường nên phải chắc chắn nó có mặt.
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
        return None, None
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        llm = _ragas_compat_gemini()(model=judge_model or GEMINI_JUDGE_MODEL,
                                     google_api_key=GOOGLE_API_KEY,
                                     temperature=0.0)
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL,
                                                  google_api_key=GOOGLE_API_KEY)
        return llm, embeddings
    return None, None


def answer_from_context(question: str, contexts: list[str]) -> str:
    """Sinh câu trả lời grounded trên context (dùng chung cho naive baseline + production)."""
    if not contexts:
        return "Không tìm thấy thông tin."

    context_str = ("\n\n---\n\n").join(contexts)
    system = ("Bạn là trợ lý tra cứu chính sách nội bộ. Trả lời CHỈ dựa trên context được cung cấp. "
              "Trả lời ngắn gọn, trực tiếp vào câu hỏi, nêu rõ con số/điều kiện nếu có. "
              "Nếu context không chứa thông tin → trả lời đúng một câu: 'Không tìm thấy.'")
    user = f"Context:\n{context_str}\n\nCâu hỏi: {question}"

    answer = chat(system, user, max_tokens=400)
    return answer if answer else contexts[0]
