import React from 'react';
import { Mic, Radio } from 'lucide-react';

export default function LiveMicTile({ onStreamStart }) {

    return (
        <div className="glass-panel" style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            borderColor: 'var(--border-subtle)'
        }}>
            <div className="panel-header">
                <Radio size={20} className="panel-icon" color="var(--text-secondary)" />
                <h3 className="panel-title">Live API Streaming</h3>
            </div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem', lineHeight: '1.4' }}>
                Capture microphone packets and stream to the remote CMAG WebSocket engine in real-time chunks.
            </p>

            <div style={{ marginTop: 'auto' }}>
                <button className="btn-primary" onClick={onStreamStart}>
                    <Mic size={20} /> Start Live Microphone
                </button>
            </div>
        </div>
    );
}
