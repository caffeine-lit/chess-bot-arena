import html
import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.svg
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

PAD = "<PAD>"
UNK = "<UNK>"
WINNER_OPTIONS = ["W", "B", "D", "?"]
MODEL_DIR = Path("models")
PIECE_ASSET_DIR = Path(__file__).parent / "assets" / "pieces" / "cburnett"
SVG_BOARD_COMPONENT_DIR = Path(__file__).parent / "components" / "svg_drag_board"
svg_drag_board_component = components.declare_component(
    "svg_drag_board", path=str(SVG_BOARD_COMPONENT_DIR)
)


class NextMoveLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        winner_embed_dim: int = 8,
        use_winner: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.use_winner = use_winner
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.winner_embed = nn.Embedding(4, winner_embed_dim)
        classifier_in = hidden_dim + (winner_embed_dim if use_winner else 0)
        self.classifier = nn.Linear(classifier_in, vocab_size)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, winner_ids: torch.Tensor) -> torch.Tensor:
        emb = self.token_embed(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        last_hidden = h[-1]
        if self.use_winner:
            w_emb = self.winner_embed(winner_ids)
            x = torch.cat([last_hidden, w_emb], dim=-1)
        else:
            x = last_hidden
        return self.classifier(x)


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get(UNK, 1)
    return [vocab.get(tok, unk) for tok in tokens]


def winner_to_id(side: str) -> int:
    return {"W": 0, "B": 1, "D": 2, "?": 3}.get(side, 3)


def best_legal_from_topk(topk_tokens: List[str], context: List[str]) -> str:
    board = board_from_context(context)
    for tok in topk_tokens:
        try:
            mv = chess.Move.from_uci(tok)
        except Exception:
            continue
        if mv in board.legal_moves:
            return tok
    return ""


@dataclass
class PlayConfig:
    winner_side: str = "B"
    topk: int = 10
    user_color: str = "white"


class LoadedMoveModel:
    def __init__(self, artifact: Dict):
        self.artifact = artifact
        self.vocab = artifact["vocab"]
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        cfg = artifact["config"]
        self.model = NextMoveLSTM(vocab_size=len(self.vocab), **cfg)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.to(torch.device("cpu"))
        self.model.eval()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "LoadedMoveModel":
        artifact = torch.load(io.BytesIO(payload), map_location="cpu")
        return cls(artifact)

    @classmethod
    def from_path(cls, model_path: Path) -> "LoadedMoveModel":
        artifact = torch.load(str(model_path), map_location="cpu")
        return cls(artifact)

    def infer(self, context: List[str], winner_side: str, topk: int) -> Dict:
        context_ids = encode_tokens(context, self.vocab)
        if not context_ids:
            context_ids = [self.vocab.get(UNK, 1)]
        tokens = torch.tensor([context_ids], dtype=torch.long)
        lengths = torch.tensor([len(context_ids)], dtype=torch.long)
        winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(tokens, lengths, winners)
            k = max(1, min(int(topk), logits.shape[-1]))
            pred_ids = logits.topk(k, dim=1).indices[0].tolist()

        topk_tokens = [self.inv_vocab.get(i, "") for i in pred_ids]
        return {
            "topk": topk_tokens,
            "best_legal": best_legal_from_topk(topk_tokens, context),
        }


def board_from_context(context: List[str]) -> chess.Board:
    board = chess.Board()
    for i, uci in enumerate(context, start=1):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception as exc:
            raise ValueError(f"Invalid UCI at ply {i}: {uci}") from exc
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal context move at ply {i}: {uci}")
        board.push(mv)
    return board


def move_list_with_san(context: List[str]) -> List[Dict[str, str]]:
    board = chess.Board()
    out: List[Dict[str, str]] = []
    for ply, uci in enumerate(context, start=1):
        mv = chess.Move.from_uci(uci)
        san = board.san(mv)
        board.push(mv)
        out.append({"ply": str(ply), "uci": uci, "san": san})
    return out


def normalize_user_move(board: chess.Board, uci: str) -> chess.Move:
    try:
        mv = chess.Move.from_uci(uci)
    except Exception as exc:
        raise ValueError(f"Invalid move format: {uci}") from exc
    if mv in board.legal_moves:
        return mv
    if len(uci) == 4:
        try_q = chess.Move.from_uci(uci + "q")
        if try_q in board.legal_moves:
            return try_q
    raise ValueError(f"Illegal move: {uci}")


def apply_user_and_model_move(
    model_runtime: LoadedMoveModel,
    context: List[str],
    user_move_uci: str,
    cfg: PlayConfig,
) -> Dict:
    board = board_from_context(context)
    expected_user_turn = chess.WHITE if cfg.user_color == "white" else chess.BLACK
    if board.turn != expected_user_turn:
        raise ValueError(f"It is not {cfg.user_color}'s turn")

    user_move = normalize_user_move(board, user_move_uci)
    user_uci = user_move.uci()
    user_san = board.san(user_move)
    board.push(user_move)
    next_context = context + [user_uci]

    model_reply: Optional[Dict] = None
    if not board.is_game_over(claim_draw=True):
        infer = model_runtime.infer(next_context, winner_side=cfg.winner_side, topk=cfg.topk)
        reply_uci = infer.get("best_legal", "")
        if reply_uci:
            reply_move = chess.Move.from_uci(reply_uci)
            if reply_move in board.legal_moves:
                reply_san = board.san(reply_move)
                board.push(reply_move)
                next_context.append(reply_uci)
                model_reply = {
                    "uci": reply_uci,
                    "san": reply_san,
                    "topk": infer.get("topk", []),
                }
            else:
                model_reply = {
                    "uci": "",
                    "san": "",
                    "topk": infer.get("topk", []),
                    "error": "predicted move not legal",
                }
        else:
            model_reply = {
                "uci": "",
                "san": "",
                "topk": infer.get("topk", []),
                "error": "no legal model move",
            }

        if (not model_reply or not model_reply.get("uci")) and not board.is_game_over(claim_draw=True):
            fallback_move = next(iter(board.legal_moves), None)
            if fallback_move is not None:
                fallback_san = board.san(fallback_move)
                board.push(fallback_move)
                next_context.append(fallback_move.uci())
                model_reply = {
                    "uci": fallback_move.uci(),
                    "san": fallback_san,
                    "topk": (model_reply or {}).get("topk", []),
                    "fallback": True,
                    "error": (model_reply or {}).get("error", "") or "model fallback used",
                }

    final_board = board_from_context(next_context)
    return {
        "context": next_context,
        "fen": final_board.fen(),
        "turn": "white" if final_board.turn == chess.WHITE else "black",
        "game_over": final_board.is_game_over(claim_draw=True),
        "result": final_board.result(claim_draw=True) if final_board.is_game_over(claim_draw=True) else "*",
        "is_check": final_board.is_check(),
        "moves": move_list_with_san(next_context),
        "last_user_move": {"uci": user_uci, "san": user_san},
        "last_model_move": model_reply,
    }


def find_local_models() -> List[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted([p for p in MODEL_DIR.rglob("*.pt") if p.is_file()], key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)


def latest_local_model() -> Optional[Path]:
    models = find_local_models()
    return models[0] if models else None


def ensure_session_state() -> None:
    ss = st.session_state
    ss.setdefault("context", [])
    ss.setdefault("last_error", "")
    ss.setdefault("last_info", "")
    ss.setdefault("last_topk", [])
    ss.setdefault("last_model_move", None)
    ss.setdefault("last_user_move", None)
    ss.setdefault("move_select", None)
    ss.setdefault("loaded_model_key", "")
    ss.setdefault("loaded_model_name", "")
    ss.setdefault("model_runtime", None)
    ss.setdefault("uploaded_model_bytes", None)
    ss.setdefault("uploaded_model_name", "")
    ss.setdefault("use_uploaded_model", False)
    ss.setdefault("click_from_square", "")
    ss.setdefault("last_svg_drag_event_id", "")
    ss.setdefault("score_w", 0)
    ss.setdefault("score_b", 0)
    ss.setdefault("score_d", 0)
    ss.setdefault("last_completed_result", "")


@st.cache_resource(show_spinner=False)
def load_model_from_path_cached(path_str: str, mtime_ns: int, size: int) -> LoadedMoveModel:
    del mtime_ns, size
    return LoadedMoveModel.from_path(Path(path_str))


@st.cache_resource(show_spinner=False)
def load_model_from_bytes_cached(payload: bytes, digest: str) -> LoadedMoveModel:
    del digest
    return LoadedMoveModel.from_bytes(payload)


@st.cache_data(show_spinner=False)
def load_piece_asset_data_urls() -> Dict[str, str]:
    piece_map: Dict[str, str] = {}
    symbols = ["P", "N", "B", "R", "Q", "K"]
    for sym in symbols:
        white_path = PIECE_ASSET_DIR / f"w{sym}.svg"
        black_path = PIECE_ASSET_DIR / f"b{sym}.svg"
        if white_path.exists():
            piece_map[sym] = "data:image/svg+xml;base64," + base64.b64encode(white_path.read_bytes()).decode("ascii")
        if black_path.exists():
            piece_map[sym.lower()] = "data:image/svg+xml;base64," + base64.b64encode(black_path.read_bytes()).decode("ascii")
    return piece_map


def load_active_model_from_sidebar() -> Tuple[Optional[LoadedMoveModel], str]:
    ss = st.session_state
    uploaded = st.sidebar.file_uploader("Upload .pt model", type=["pt"], help="Optional: overrides local model selection")
    if uploaded is not None:
        payload = uploaded.getvalue()
        ss["uploaded_model_bytes"] = payload
        ss["uploaded_model_name"] = uploaded.name
        ss["use_uploaded_model"] = True

    use_uploaded = st.sidebar.toggle(
        "Use uploaded model",
        value=bool(ss.get("use_uploaded_model") and ss.get("uploaded_model_bytes")),
        disabled=not bool(ss.get("uploaded_model_bytes")),
    )
    ss["use_uploaded_model"] = use_uploaded

    local_models = find_local_models()
    local_model_labels = [str(p) for p in local_models]
    default_local = str(latest_local_model()) if latest_local_model() else "models/model.pt"
    if local_model_labels:
        if default_local in local_model_labels:
            default_idx = local_model_labels.index(default_local)
        else:
            default_idx = 0
        selected_local = st.sidebar.selectbox("Local model", local_model_labels, index=default_idx)
    else:
        selected_local = st.sidebar.text_input("Local model path", value=default_local)

    if use_uploaded and ss.get("uploaded_model_bytes"):
        payload = ss["uploaded_model_bytes"]
        digest = hashlib.sha256(payload).hexdigest()
        key = f"upload:{digest}"
        if ss.get("loaded_model_key") != key or ss.get("model_runtime") is None:
            with st.spinner("Loading uploaded model..."):
                ss["model_runtime"] = load_model_from_bytes_cached(payload, digest)
                ss["loaded_model_key"] = key
                ss["loaded_model_name"] = ss.get("uploaded_model_name", "uploaded.pt")
        return ss["model_runtime"], ss["loaded_model_name"]

    path = Path(selected_local)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        st.sidebar.warning(f"Model not found: {path}")
        return None, str(path)

    stat = path.stat()
    key = f"path:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    if ss.get("loaded_model_key") != key or ss.get("model_runtime") is None:
        with st.spinner("Loading model..."):
            ss["model_runtime"] = load_model_from_path_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            ss["loaded_model_key"] = key
            ss["loaded_model_name"] = str(path)
    return ss["model_runtime"], ss["loaded_model_name"]


def current_board() -> chess.Board:
    return board_from_context(st.session_state["context"])


def legal_move_options(board: chess.Board) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for mv in board.legal_moves:
        san = board.san(mv)
        options.append((mv.uci(), san))
    options.sort(key=lambda x: x[1])
    return options


def reset_game() -> None:
    try:
        board = board_from_context(st.session_state.get("context", []))
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            if result == "1-0":
                st.session_state["score_w"] = int(st.session_state.get("score_w", 0)) + 1
            elif result == "0-1":
                st.session_state["score_b"] = int(st.session_state.get("score_b", 0)) + 1
            elif result == "1/2-1/2":
                st.session_state["score_d"] = int(st.session_state.get("score_d", 0)) + 1
            st.session_state["last_completed_result"] = result
    except Exception:
        pass

    st.session_state["context"] = []
    st.session_state["last_error"] = ""
    st.session_state["last_info"] = "New game started."
    st.session_state["last_topk"] = []
    st.session_state["last_model_move"] = None
    st.session_state["last_user_move"] = None
    st.session_state["click_from_square"] = ""


def undo_pair() -> None:
    ctx = st.session_state["context"]
    if not ctx:
        st.session_state["last_error"] = "No moves to undo."
        return
    st.session_state["context"] = ctx[:-2] if len(ctx) >= 2 else []
    st.session_state["last_error"] = ""
    st.session_state["last_info"] = "Undid last user+AI pair (or reset if fewer than 2 plies)."
    st.session_state["last_topk"] = []
    st.session_state["last_model_move"] = None
    st.session_state["last_user_move"] = None
    st.session_state["click_from_square"] = ""


def play_selected_move(model_runtime: LoadedMoveModel, cfg: PlayConfig, user_move_uci: str) -> None:
    try:
        result = apply_user_and_model_move(model_runtime, st.session_state["context"], user_move_uci, cfg)
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state["last_info"] = ""
        return

    st.session_state["context"] = result["context"]
    st.session_state["last_error"] = ""
    st.session_state["last_user_move"] = result.get("last_user_move")
    st.session_state["last_model_move"] = result.get("last_model_move")
    st.session_state["click_from_square"] = ""
    model_reply = result.get("last_model_move") or {}
    st.session_state["last_topk"] = model_reply.get("topk", [])
    parts = []
    if result.get("last_user_move"):
        parts.append(f"You: {result['last_user_move']['san']}")
    if model_reply.get("san"):
        suffix = " (fallback)" if model_reply.get("fallback") else ""
        parts.append(f"AI: {model_reply['san']}{suffix}")
    if model_reply.get("error"):
        parts.append(f"Note: {model_reply['error']}")
    st.session_state["last_info"] = " | ".join(parts)


def render_board(board: chess.Board, user_color: str = "white") -> None:
    lastmove = None
    if st.session_state["context"]:
        try:
            lastmove = chess.Move.from_uci(st.session_state["context"][-1])
        except Exception:
            lastmove = None
    orientation = chess.WHITE if user_color == "white" else chess.BLACK
    svg = chess.svg.board(board=board, size=520, lastmove=lastmove, orientation=orientation)
    components.html(f"<div style='max-width:520px'>{svg}</div>", height=540, scrolling=False)


def last_context_move_uci() -> str:
    ctx = st.session_state.get("context") or []
    return ctx[-1] if ctx else ""


def render_draggable_svg_board(board: chess.Board, user_color: str, disabled: bool) -> Optional[str]:
    payload = svg_drag_board_component(
        fen=board.fen(),
        orientation=user_color,
        disabled=bool(disabled),
        lastmove_uci=last_context_move_uci(),
        piece_assets=load_piece_asset_data_urls(),
        key="svg_drag_board_main",
        default=None,
    )
    if not isinstance(payload, dict):
        return None
    event_id = str(payload.get("event_id", ""))
    move_uci = str(payload.get("move_uci", "")).strip().lower()
    if not event_id or not move_uci:
        return None
    if st.session_state.get("last_svg_drag_event_id") == event_id:
        return None
    st.session_state["last_svg_drag_event_id"] = event_id
    return move_uci


PIECE_GLYPHS = {
    "P": "P",
    "N": "N",
    "B": "B",
    "R": "R",
    "Q": "Q",
    "K": "K",
    "p": "p",
    "n": "n",
    "b": "b",
    "r": "r",
    "q": "q",
    "k": "k",
}


def square_button_label(board: chess.Board, square: chess.Square, selected_from: str, legal_dests: set[str]) -> str:
    sq_name = chess.square_name(square)
    piece = board.piece_at(square)
    piece_text = PIECE_GLYPHS.get(piece.symbol(), ".") if piece else "."
    marker = ""
    if sq_name == selected_from:
        marker = "*"
    elif sq_name in legal_dests:
        marker = "+"
    return f"{sq_name} {piece_text}{marker}"


def render_mouse_move_picker(board: chess.Board, user_color: str) -> Optional[str]:
    st.caption("Mouse move input: click a piece square, then click a destination square.")
    selected_from = st.session_state.get("click_from_square", "")

    legal_dests: set[str] = set()
    if selected_from:
        try:
            from_sq = chess.parse_square(selected_from)
            for mv in board.legal_moves:
                if mv.from_square == from_sq:
                    legal_dests.add(chess.square_name(mv.to_square))
        except Exception:
            st.session_state["click_from_square"] = ""
            selected_from = ""

    info_cols = st.columns([1, 1])
    info_cols[0].write(f"Selected: `{selected_from or '-'} `")
    if info_cols[1].button("Clear Selection", use_container_width=True):
        st.session_state["click_from_square"] = ""
        return None

    files = list("abcdefgh")
    ranks = list(range(8, 0, -1))
    if user_color == "black":
        files = list(reversed(files))
        ranks = list(range(1, 9))

    clicked_square: Optional[str] = None
    for rank in ranks:
        cols = st.columns(8)
        for i, file_char in enumerate(files):
            sq_name = f"{file_char}{rank}"
            sq = chess.parse_square(sq_name)
            label = square_button_label(board, sq, selected_from, legal_dests)
            if cols[i].button(label, key=f"sqbtn_{sq_name}", use_container_width=True):
                clicked_square = sq_name

    if not clicked_square:
        return None

    piece = board.piece_at(chess.parse_square(clicked_square))
    if not selected_from:
        if piece is None:
            st.session_state["last_error"] = "Select one of your pieces first."
            return None
        if piece.color != board.turn:
            st.session_state["last_error"] = "Select a piece for the side to move."
            return None
        st.session_state["click_from_square"] = clicked_square
        st.session_state["last_error"] = ""
        return None

    if clicked_square == selected_from:
        st.session_state["click_from_square"] = ""
        st.session_state["last_error"] = ""
        return None

    candidate_uci = selected_from + clicked_square
    try:
        normalized = normalize_user_move(board, candidate_uci)
    except ValueError:
        if piece is not None and piece.color == board.turn:
            st.session_state["click_from_square"] = clicked_square
            st.session_state["last_error"] = ""
            return None
        st.session_state["last_error"] = f"Illegal move: {candidate_uci}"
        return None

    st.session_state["click_from_square"] = ""
    st.session_state["last_error"] = ""
    return normalized.uci()


def render_move_history() -> None:
    rows = move_list_with_san(st.session_state["context"])
    if not rows:
        st.caption("No moves yet.")
        return
    for i in range(0, len(rows), 2):
        white = rows[i]
        black = rows[i + 1] if i + 1 < len(rows) else None
        move_no = (i // 2) + 1
        line = f"{move_no}. {white['san']}"
        if black:
            line += f"   {black['san']}"
        st.text(line)


def move_history_lines() -> List[str]:
    rows = move_list_with_san(st.session_state["context"])
    lines: List[str] = []
    for i in range(0, len(rows), 2):
        white = rows[i]
        black = rows[i + 1] if i + 1 < len(rows) else None
        move_no = (i // 2) + 1
        line = f"{move_no}. {white['san']}"
        if black:
            line += f"   {black['san']}"
        lines.append(line)
    lines.reverse()
    return lines


def render_move_history_scrollbox(height_px: int = 220) -> None:
    lines = move_history_lines()
    if not lines:
        st.caption("No moves yet.")
        return
    body = "<br>".join(html.escape(line) for line in lines)
    st.markdown(
        (
            f"<div style='max-height:{height_px}px; overflow-y:auto; border:1px solid #e5e7eb; "
            "border-radius:8px; padding:8px 10px; background:#fff; font-family:monospace; "
            "white-space:pre-wrap;'>"
            f"{body}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_message_box(error_text: str, info_text: str) -> None:
    if error_text:
        bg = "#fbe4e6"
        fg = "#7a2730"
        text = html.escape(error_text)
    elif info_text:
        bg = "#e7f0fb"
        fg = "#163a63"
        text = html.escape(info_text)
    else:
        bg = "#f8fafc"
        fg = "#64748b"
        text = "&nbsp;"
    st.markdown(
        (
            "<div style='min-height:52px; display:flex; align-items:center; "
            f"padding:8px 12px; border-radius:8px; background:{bg}; color:{fg}; "
            "margin: 0.25rem 0 0.5rem 0;'>"
            f"{text}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_topk_row(tokens: List[str]) -> None:
    if tokens:
        chips = " ".join(
            f"<code>{idx}. {html.escape(tok)}</code>" for idx, tok in enumerate(tokens, start=1)
        )
        body = chips
    else:
        body = "<span style='color:#6b7280'>No AI move yet.</span>"
    st.markdown(
        (
            "<div style='min-height:44px; display:flex; align-items:flex-start; "
            "flex-wrap:wrap; gap:6px; line-height:1.8;'>"
            f"{body}</div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Chess AI Streamlit Demo", layout="wide")
    ensure_session_state()
    st.markdown(
        """
        <style>
        .stApp .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1, h2, h3 {
            margin-top: 0.25rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Play Chess Against a Pretrained Model")
    st.caption("Standalone Streamlit demo with swappable `.pt` models (winner-aware next-move predictor, CPU inference).")

    with st.sidebar:
        st.header("Model")
        st.caption("Place models in `models/` or upload a `.pt` file.")
    model_runtime, model_name = load_active_model_from_sidebar()

    st.sidebar.header("Game Config")
    winner_side = st.sidebar.selectbox("winner_side token", WINNER_OPTIONS, index=WINNER_OPTIONS.index("B"))
    topk = st.sidebar.slider("top-k candidates", min_value=1, max_value=50, value=10)
    user_color = st.sidebar.selectbox("Your color", ["white", "black"], index=0)
    if user_color != "white":
        st.sidebar.info("Current UI is optimized for playing as White. Black works only if it is Black to move.")

    col_a, col_b = st.columns([1.15, 1.0])
    cfg = PlayConfig(winner_side=winner_side, topk=int(topk), user_color=user_color)

    try:
        board = current_board()
    except Exception as exc:
        st.session_state["context"] = []
        st.session_state["last_error"] = f"Resetting invalid game state: {exc}"
        board = chess.Board()

    with col_a:
        st.subheader("Board")
        user_turn_now = (board.turn == chess.WHITE and user_color == "white") or (
            board.turn == chess.BLACK and user_color == "black"
        )
        drag_disabled = board.is_game_over(claim_draw=True) or (not user_turn_now) or (model_runtime is None)
        dragged_move = render_draggable_svg_board(board, user_color=user_color, disabled=drag_disabled)
        if dragged_move:
            if model_runtime is None:
                st.session_state["last_error"] = "Load a model first."
            else:
                play_selected_move(model_runtime, cfg, dragged_move)
            st.rerun()
        st.caption("Drag a piece on the board to move. Promotion defaults to queen.")
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        if ctrl1.button("New Game", use_container_width=True):
            reset_game()
            st.rerun()
        if ctrl2.button("Undo Pair", use_container_width=True):
            undo_pair()
            st.rerun()
        if ctrl3.button("Copy FEN to Status", use_container_width=True):
            st.session_state["last_info"] = board.fen()
            st.session_state["last_error"] = ""
            st.rerun()

        st.subheader("Your Move")
        if board.is_game_over(claim_draw=True):
            st.success(f"Game over: {board.result(claim_draw=True)}")
        elif (board.turn == chess.WHITE and user_color == "white") or (board.turn == chess.BLACK and user_color == "black"):
            options = legal_move_options(board)
            if not options:
                st.warning("No legal moves available.")
            else:
                with st.expander("Fallback: move list selector"):
                    option_map = {uci: san for uci, san in options}
                    move_keys = [uci for uci, _ in options]
                    default_idx = 0
                    current_sel = st.session_state.get("move_select")
                    if current_sel in option_map:
                        default_idx = move_keys.index(current_sel)
                    selected_uci = st.selectbox(
                        "Choose a move",
                        move_keys,
                        index=default_idx,
                        format_func=lambda uci: f"{option_map[uci]} ({uci})",
                        key="move_select",
                    )
                    if st.button("Play Selected Move", type="primary", use_container_width=True, disabled=model_runtime is None):
                        if model_runtime is None:
                            st.session_state["last_error"] = "Load a model first."
                        else:
                            play_selected_move(model_runtime, cfg, selected_uci)
                        st.rerun()
        else:
            st.info("It is the AI turn based on selected color. Adjust color or start a new game.")

    with col_b:
        st.subheader("Status")
        st.write(f"Model: `{model_name}`" if model_name else "Model: not loaded")
        st.write(f"Turn: `{'white' if board.turn == chess.WHITE else 'black'}`")
        st.write(f"Result: `{board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else '*'}`")
        st.write(f"Check: `{board.is_check()}`")
        st.write(
            "Session score: "
            f"`W {st.session_state.get('score_w', 0)} / "
            f"B {st.session_state.get('score_b', 0)} / "
            f"D {st.session_state.get('score_d', 0)}`"
        )
        if st.session_state.get("last_completed_result"):
            st.caption(f"Last completed game: {st.session_state['last_completed_result']}")

        render_status_message_box(
            st.session_state.get("last_error", ""),
            st.session_state.get("last_info", ""),
        )

        st.subheader("Move History")
        render_move_history_scrollbox()

        st.subheader("Last AI Top-k")
        topk_tokens = st.session_state.get("last_topk") or []
        render_topk_row(topk_tokens)

        with st.expander("Current Context (UCI list)", expanded=True):
            st.code(" ".join(st.session_state["context"]) or "<empty>")

        with st.expander("Deployment Notes"):
            st.markdown(
                "- Put a compatible `.pt` artifact in `models/` or upload one in the sidebar.\n"
                "- This demo runs inference on CPU only (`torch.load(..., map_location='cpu')` + model moved to CPU).\n"
                "- If the model predicts no legal move, the app plays a legal fallback move so the game continues."
            )


if __name__ == "__main__":
    main()
