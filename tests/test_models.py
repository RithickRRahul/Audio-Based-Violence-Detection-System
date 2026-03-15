import pytest
import torch
import numpy as np
from src.models.audio_encoder import AudioEncoder
from src.models.nlp_encoder import TextEncoder
from src.models.scream_detector import ScreamDetector
from src.models.cmag_v2 import EnhancedCMAG
from src.models.temporal import TemporalEscalation
from src.config import N_MELS, AUDIO_EMBED_DIM, TEXT_EMBED_DIM, FUSION_HIDDEN_DIM


# ============================================================
# Phase 2: Audio Encoder Tests (passing)
# ============================================================

def test_audio_encoder_output_shape():
    batch_size = 4
    time_steps = 126
    dummy_input = torch.randn(batch_size, 1, N_MELS, time_steps)
    model = AudioEncoder()
    output = model(dummy_input)
    assert output.shape == (batch_size, AUDIO_EMBED_DIM)

def test_audio_encoder_variable_length():
    batch_size = 2
    model = AudioEncoder()
    model.eval()
    out_short = model(torch.randn(batch_size, 1, N_MELS, 60))
    out_long = model(torch.randn(batch_size, 1, N_MELS, 200))
    assert out_short.shape == (batch_size, AUDIO_EMBED_DIM)
    assert out_long.shape == (batch_size, AUDIO_EMBED_DIM)


# ============================================================
# Phase 3: NLP Pipeline Tests (passing)
# ============================================================

def test_text_encoder_output_shape():
    model = TextEncoder()
    texts = ["This is a threat", "Hello how are you"]
    embeddings = model.get_embeddings(texts)
    assert embeddings.shape == (2, TEXT_EMBED_DIM)

def test_text_encoder_threat_score():
    model = TextEncoder()
    score = model.get_threat_score("I am going to hurt you")
    assert 0.0 <= score <= 1.0

def test_text_encoder_empty_string():
    model = TextEncoder()
    embeddings = model.get_embeddings([""])
    assert embeddings.shape == (1, TEXT_EMBED_DIM)

def test_scream_detector_detects_screaming_text():
    detector = ScreamDetector()
    result = detector.detect("AAAAHHHHH HELP ME PLEASE")
    assert result is True

def test_scream_detector_safe_text():
    detector = ScreamDetector()
    result = detector.detect("Hello, how are you doing today?")
    assert result is False

def test_scream_detector_acoustic():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    scream_audio = np.sin(2 * np.pi * 2500 * t) * 0.8
    detector = ScreamDetector()
    result = detector.detect_acoustic(scream_audio, sr)
    assert isinstance(result, bool)

def test_albert_fusion_dimensionality():
    encoder = TextEncoder(model_name="albert-base-v2")
    embeds, scores = encoder(["He hit me hard!"])
    
    # TextEncoder now uses purely the transformer hidden dim = 768
    assert embeds.shape == (1, 768)
    assert 0.0 <= torch.sigmoid(scores).item() <= 1.0

def test_bert_fusion_dimensionality():
    """
    Test driven development implementation verifying bert-base-uncased
    can successfully route through correctly.
    """
    encoder = TextEncoder(model_name="bert-base-uncased")
    embeds, scores = encoder(["I will completely destroy your life"])
    
    # BERT base hidden dim is 768.
    assert embeds.shape == (1, 768)
    assert 0.0 <= torch.sigmoid(scores).item() <= 1.0


# ============================================================
# Phase 4: Enhanced CMAG-v2 Fusion Tests (passing)
# ============================================================

def test_cmag_v2_output_shape():
    batch_size = 4
    audio_emb = torch.randn(batch_size, AUDIO_EMBED_DIM)
    text_emb = torch.randn(batch_size, TEXT_EMBED_DIM)
    model = EnhancedCMAG()
    output = model(audio_emb, text_emb)
    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_cmag_v2_returns_segment_features():
    batch_size = 4
    audio_emb = torch.randn(batch_size, AUDIO_EMBED_DIM)
    text_emb = torch.randn(batch_size, TEXT_EMBED_DIM)
    model = EnhancedCMAG()
    output, fused_features = model(audio_emb, text_emb, return_features=True)
    assert output.shape == (batch_size, 1)
    assert fused_features.shape == (batch_size, FUSION_HIDDEN_DIM * 2)

def test_cmag_v2_gating_mechanism():
    batch_size = 2
    audio_emb = torch.randn(batch_size, AUDIO_EMBED_DIM)
    text_emb = torch.zeros(batch_size, TEXT_EMBED_DIM)
    model = EnhancedCMAG()
    model.eval()
    output = model(audio_emb, text_emb)
    assert not torch.isnan(output).any()
    assert output.shape == (batch_size, 1)

def test_cmag_v2_bidirectional_gates():
    batch_size = 2
    audio_emb = torch.randn(batch_size, AUDIO_EMBED_DIM)
    text_emb = torch.randn(batch_size, TEXT_EMBED_DIM)
    model = EnhancedCMAG()
    model.eval()
    out1 = model(audio_emb, text_emb)
    out2 = model(audio_emb, text_emb)
    assert torch.allclose(out1, out2)


# ============================================================
# Phase 5: Temporal Escalation Tests — EXTRA FOCUS
# ============================================================

def test_temporal_output_shape():
    """
    Test basic output shape.
    Input: (B, seq_len, fusion_dim) — sequence of segment features from CMAG
    Output: (B, 1) — file-level violence probability
    """
    batch_size = 2
    seq_len = 5  # 5 segments = 20 seconds
    fusion_dim = FUSION_HIDDEN_DIM * 2  # 256

    model = TemporalEscalation(input_dim=fusion_dim)
    dummy_seq = torch.randn(batch_size, seq_len, fusion_dim)
    output = model(dummy_seq)

    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_temporal_variable_sequence_length():
    """
    Test that the Bi-LSTM handles arbitrary sequence lengths.
    Critical for real-world: audio files can be 10s, 60s, or 5 minutes.
    """
    batch_size = 2
    fusion_dim = FUSION_HIDDEN_DIM * 2
    model = TemporalEscalation(input_dim=fusion_dim)
    model.eval()

    out_short = model(torch.randn(batch_size, 3, fusion_dim))   # 12 seconds
    out_medium = model(torch.randn(batch_size, 15, fusion_dim))  # 60 seconds
    out_long = model(torch.randn(batch_size, 75, fusion_dim))    # 5 minutes

    assert out_short.shape == (batch_size, 1)
    assert out_medium.shape == (batch_size, 1)
    assert out_long.shape == (batch_size, 1)

def test_temporal_per_segment_scores():
    """
    Test that the model can return per-segment violence scores
    (not just the overall prediction). This is crucial for the
    web UI which needs to show timestamp-level results.
    """
    batch_size = 2
    seq_len = 5
    fusion_dim = FUSION_HIDDEN_DIM * 2

    model = TemporalEscalation(input_dim=fusion_dim)
    dummy_seq = torch.randn(batch_size, seq_len, fusion_dim)
    output, per_segment = model(dummy_seq, return_per_segment=True)

    assert output.shape == (batch_size, 1)
    assert per_segment.shape == (batch_size, seq_len, 1)
    assert torch.all(per_segment >= 0) and torch.all(per_segment <= 1)

def test_temporal_attention_weights():
    """
    Test that the self-attention mechanism produces valid attention weights.
    Attention weights should sum to 1 across the sequence dimension.
    This proves the model is LEARNING which segments matter most.
    """
    batch_size = 2
    seq_len = 5
    fusion_dim = FUSION_HIDDEN_DIM * 2

    model = TemporalEscalation(input_dim=fusion_dim)
    dummy_seq = torch.randn(batch_size, seq_len, fusion_dim)
    output, attn_weights = model(dummy_seq, return_attention=True)

    assert output.shape == (batch_size, 1)
    assert attn_weights.shape == (batch_size, seq_len)
    # Attention weights should sum to ~1 (softmax)
    weight_sums = attn_weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5)

def test_temporal_single_segment():
    """
    Edge case: only 1 segment (4 seconds of audio).
    The model should still produce a valid prediction.
    """
    batch_size = 2
    fusion_dim = FUSION_HIDDEN_DIM * 2
    model = TemporalEscalation(input_dim=fusion_dim)
    model.eval()

    output = model(torch.randn(batch_size, 1, fusion_dim))
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()

def test_temporal_deterministic_in_eval():
    """
    Test that the model is deterministic in eval mode.
    """
    batch_size = 2
    seq_len = 5
    fusion_dim = FUSION_HIDDEN_DIM * 2

    model = TemporalEscalation(input_dim=fusion_dim)
    model.eval()
    dummy_seq = torch.randn(batch_size, seq_len, fusion_dim)

    out1 = model(dummy_seq)
    out2 = model(dummy_seq)
    assert torch.allclose(out1, out2)

def test_temporal_gradient_flow():
    """
    Test that gradients flow through the entire temporal model.
    This validates that the model is actually trainable.
    Critical: previous temporal approaches may have failed due to
    vanishing gradients in naive LSTM implementations.
    """
    batch_size = 2
    seq_len = 5
    fusion_dim = FUSION_HIDDEN_DIM * 2

    model = TemporalEscalation(input_dim=fusion_dim)
    model.train()
    dummy_seq = torch.randn(batch_size, seq_len, fusion_dim, requires_grad=True)

    # Test main path (attention-based classification)
    output = model(dummy_seq)
    loss = output.sum()
    loss.backward()

    # Gradients should exist and be non-zero for the input
    assert dummy_seq.grad is not None
    assert dummy_seq.grad.abs().sum() > 0

    # Check gradients for main path parameters
    # (segment_scorer only gets gradients when return_per_segment=True)
    main_path_prefixes = ("input_norm", "lstm", "attention", "classifier")
    for name, param in model.named_parameters():
        if param.requires_grad and any(name.startswith(p) for p in main_path_prefixes):
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    # Test per-segment path separately
    model.zero_grad()
    dummy_seq2 = torch.randn(batch_size, seq_len, fusion_dim, requires_grad=True)
    output2, per_seg = model(dummy_seq2, return_per_segment=True)
    loss2 = (output2.sum() + per_seg.sum())
    loss2.backward()

    # Now segment_scorer should also have gradients
    for name, param in model.named_parameters():
        if param.requires_grad and name.startswith("segment_scorer"):
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
